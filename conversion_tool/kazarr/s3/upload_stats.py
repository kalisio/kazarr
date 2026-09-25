"""
kazarr.s3.upload_stats — optional, opt-in diagnostic instrumentation for S3
chunk uploads: for every PutObject/UploadPart *attempt*, records its size,
wall-clock duration, HTTP status, S3 error code (if any) and attempt number
to a JSON-lines file.

Why
---
To find out whether the "IncompleteBody" failures Kazarr sometimes hits
happen at random, or systematically around a particular duration -- the
signature of a proxy or load balancer's own idle/request timeout sitting
somewhere between Kazarr and the real storage backend. Random failures with
no duration pattern point to plain network flakiness (packet loss,
connection resets); failures clustering tightly around a fixed duration
regardless of chunk size point to a timeout. See analyze_s3_upload_stats.py
for turning the recorded file into that answer.

This module is entirely self-contained: it only observes, never retries or
otherwise changes S3 behavior, and does not depend on any other Kazarr S3
module. It is disabled by default (near-zero overhead when off: one dict
lookup per S3 request). Enable it with enable_upload_stats(path) -- kazarr's
CLI wires this to --s3-upload-stats PATH.

How it works
------------
register_stats_s3_filesystem() (called once, at import time, below) makes
fsspec build a thin s3fs.S3FileSystem subclass, _StatsS3FileSystem, for
every "s3://" path -- including the one xarray/zarr build internally from
storage_options in dataset.to_zarr(). That subclass behaves exactly like
plain s3fs.S3FileSystem, except that right after it creates its underlying
aiobotocore client, it calls register_stats_handlers (this module) on it.
Two botocore events then fire once per HTTP *attempt* (an upload retried 3
times by botocore's own retry policy produces 3 records, distinguishable by
"attempt"):
  - "before-send.s3": right before an attempt is sent. Records a start
    timestamp and the request's declared size. Read off the Content-Length
    header rather than introspecting request.body -- aiobotocore wraps the
    payload in a BytesIO (or a streaming/signing wrapper) depending on
    signing, so len(body) is not reliable, but Content-Length always is
    (confirmed against aiobotocore 2.25.2 / botocore 1.40.70).
  - "needs-retry.s3": right after a response or exception is available for
    that same attempt, success or failure (botocore fires this event
    unconditionally after every attempt, regardless of whether anything
    ends up retrying it). Computes the duration and writes one record.
    Every handler registered on this event always runs, regardless of what
    any other handler (including botocore's own default retry handler)
    decides -- confirmed by reading botocore.endpoint.Endpoint._needs_retry,
    which collects ALL handler responses via plain emit() before picking
    one, unlike emit_until_response().

A contextvars.ContextVar correlates the two events for a given attempt.
This is safe under concurrent uploads (Dask writing many chunks in parallel
through the same cached client): each concurrent asyncio Task gets its own
copy of the ContextVar, so concurrent attempts never see each other's start
time or size. Verified against a local moto_server with dozens of
concurrently-uploaded objects of distinct sizes: every recorded (url, size)
pair matched its own upload, with zero cross-talk, across repeated runs.

Records are JSON lines, one per HTTP attempt. The file is opened in append
mode; each record is written as a single write() call comfortably under
PIPE_BUF (4096 bytes on Linux), so concurrent writes from multiple asyncio
tasks -- or even multiple local Dask worker *processes* pointed at the same
path -- won't interleave into a corrupt line. For a distributed multi-machine
Dask cluster, point each machine at its own path (e.g. including the
hostname) and merge the files afterwards; there is no cross-machine
coordination here.

Crossing process boundaries (important)
----------------------------------------
kazarr/processes.py's init_dask_dashboard step does `Client()` with no
arguments, and dask.distributed.LocalCluster defaults `processes=True`
whenever no worker_class is given (confirmed by reading LocalCluster.__init__:
`processes = worker_class is None or issubclass(worker_class, Nanny)`) -- so
by default Kazarr's actual chunk PutObject/UploadPart calls run in separate
Dask worker *processes* (managed by Nanny), not threads, and not the CLI's
own process. enable_upload_stats(path), called once from kazarr.cli in the
main process, only sets this module's state THERE; a worker process has its
own separate Python interpreter and its own copy of _enabled/_fd, so without
the mechanism below it would never turn stats on, and every S3 request it
issues -- i.e. essentially all of them, for anything but a trivially small
dataset -- would go unrecorded, while the file still gets created (empty) by
the CLI process's own open() call.

The fix: enable_upload_stats() also sets an environment variable
(_ENV_VAR). Dask's worker processes -- whether started via fork or spawn --
inherit the parent process's environment at the moment they're created,
which is after the CLI has already called enable_upload_stats() (the
pipeline's Client() is created later, as a pipeline step). Each worker
lazily self-enables (_ensure_enabled(), called from both event handlers
below) the first time it needs to check, opening its OWN file descriptor
onto the same path -- multiple processes appending small (<PIPE_BUF)
records to one path is safe, as above.
"""

import contextvars
import json
import logging
import os
import time
from datetime import UTC, datetime

import fsspec
import s3fs

logger = logging.getLogger(__name__)

_ENV_VAR = "KAZARR_S3_UPLOAD_STATS_PATH"

_enabled = False
_fd = None
_env_checked = False

_attempt_start = contextvars.ContextVar("kazarr_upload_stats_attempt_start", default=None)

_BEFORE_SEND_UNIQUE_ID = "kazarr-upload-stats-before-send"
_NEEDS_RETRY_UNIQUE_ID = "kazarr-upload-stats-needs-retry"

_fsspec_registered = False


def enable_upload_stats(path):
    """Start recording per-attempt S3 upload stats to `path` (JSON lines,
    appended -- an existing file is added to, not overwritten). Call once,
    before the pipeline runs; kazarr.cli wires this to --s3-upload-stats.

    Also sets an environment variable so that Dask worker *processes*
    spawned later (see this module's docstring) can each self-enable in
    their own process and open their own file descriptor onto the same
    path -- without this, only the CLI's own process would ever record
    anything, which for a real (Dask-parallelized) dataset write is next
    to nothing.
    """
    global _enabled, _fd, _env_checked
    if _fd is not None:
        _fd.close()
    _fd = open(path, "a", buffering=1)  # line-buffered # noqa: SIM115
    _enabled = True
    _env_checked = True
    os.environ[_ENV_VAR] = path
    logger.info("S3 upload stats enabled: recording per-attempt records to %s", path)


def _ensure_enabled():
    """Self-enable in a Dask worker process that inherited _ENV_VAR from
    the CLI process but never called enable_upload_stats() itself. A
    no-op (single boolean check) once this process has resolved one way
    or the other, so this costs nothing after the first S3 request in a
    given process. See this module's docstring for why this exists.
    """
    global _env_checked
    if _enabled or _env_checked:
        return
    _env_checked = True
    path = os.environ.get(_ENV_VAR)
    if path:
        enable_upload_stats(path)


def is_enabled():
    return _enabled


def _write_record(record):
    if _fd is None:
        return
    try:
        _fd.write(json.dumps(record) + "\n")
    except Exception:
        logger.debug("Failed to write S3 upload stats record", exc_info=True)


def _before_send(request, **kwargs):
    _ensure_enabled()
    if not _enabled or request.method != "PUT":
        return
    # Most PUTs carry a plain Content-Length. But when the client signs the
    # payload as aws-chunked with a trailing checksum (Transfer-Encoding:
    # chunked, Content-Encoding: aws-chunked, X-Amz-Trailer set -- e.g.
    # flexible checksums against some S3-compatible backends), there is no
    # Content-Length at all: the real (decoded) body size is instead in
    # X-Amz-Decoded-Content-Length. Confirmed against a real run targeting
    # Scaleway Object Storage, where every PUT uses this streaming-trailer
    # form -- moto (used for local testing) never exercises this path,
    # which is why this was missed initially.
    content_length = request.headers.get("Content-Length") or request.headers.get(
        "X-Amz-Decoded-Content-Length"
    )
    if not content_length:
        return
    _attempt_start.set(
        {"start": time.monotonic(), "size_bytes": int(content_length), "url": request.url}
    )
    return  # never short-circuits the actual send


def _needs_retry_stats(attempts, response, caught_exception=None, operation=None, **kwargs):
    _ensure_enabled()
    if not _enabled:
        return
    info = _attempt_start.get(None)
    if info is None:
        return
    # Consumed: clear it so an unrelated request issued later in the same
    # task (e.g. a HeadObject) doesn't inherit a stale start time.
    _attempt_start.set(None)

    duration_s = time.monotonic() - info["start"]
    status_code = None
    error_code = None
    if response is not None:
        http_response, parsed_response = response
        status_code = getattr(http_response, "status_code", None)
        if parsed_response is not None:
            error_code = parsed_response.get("Error", {}).get("Code")
    elif caught_exception is not None:
        error_code = type(caught_exception).__name__

    _write_record(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": getattr(operation, "name", None),
            "url": info["url"],
            "size_bytes": info["size_bytes"],
            "attempt": attempts,
            "duration_s": round(duration_s, 4),
            "status_code": status_code,
            "error_code": error_code,
            "success": error_code is None,
        }
    )
    return  # observation only -- never influences the retry decision


def register_stats_handlers(client):
    """Register this module's observers on an aiobotocore S3 client.

    Always registers (unique_id makes repeated calls, which s3fs makes
    before most operations, a no-op) -- the _enabled check inside the
    handlers is what makes this free when stats collection is off.
    """
    client.meta.events.register(
        "before-send.s3", _before_send, unique_id=_BEFORE_SEND_UNIQUE_ID
    )
    client.meta.events.register(
        "needs-retry.s3", _needs_retry_stats, unique_id=_NEEDS_RETRY_UNIQUE_ID
    )


class _StatsS3FileSystem(s3fs.S3FileSystem):
    """s3fs.S3FileSystem that also enables --s3-upload-stats instrumentation.

    Behaves exactly like plain s3fs.S3FileSystem in every other respect --
    no retry behavior, no other side effect -- the only difference is that
    register_stats_handlers() is called on the underlying client right
    after it is created.
    """

    async def set_session(self, refresh=False, kwargs=None):
        client = await super().set_session(refresh=refresh, kwargs=kwargs or {})
        register_stats_handlers(client)
        return client


def register_stats_s3_filesystem():
    """Make fsspec build _StatsS3FileSystem for every 's3://' path.

    Idempotent; safe to call multiple times or from multiple processes.
    Must run before the first 's3://' path is resolved through fsspec in a
    given process (kazarr.utils calls this at import time, and this module
    also calls it at its own import time as a second safety net).

    Only affects code that goes through fsspec's protocol registry
    (fsspec.filesystem(...), fsspec.url_to_fs(...), which is what
    zarr.storage.FsspecStore and xarray's to_zarr(storage_options=...) use).
    A direct `s3fs.S3FileSystem(...)` call bypasses the registry and is
    unaffected -- use `fsspec.filesystem("s3", **opts)` instead if a call
    site should benefit from this too (see get_s3_filesystem in utils.py).
    """
    global _fsspec_registered
    if _fsspec_registered:
        return
    fsspec.register_implementation("s3", _StatsS3FileSystem, clobber=True)
    _fsspec_registered = True


register_stats_s3_filesystem()


def register_stats_s3_filesystem_on_workers(client):
    """Call once, right after creating a dask.distributed.Client() (see
    kazarr/processes.py's init_dask_dashboard), so its worker *processes* --
    present, and any created later (e.g. if a crashed worker is restarted)
    -- also resolve "s3://" through _StatsS3FileSystem.

    Why this is needed
    -------------------
    register_stats_s3_filesystem() (above) runs once, at kazarr.utils
    import time, in whichever process calls it -- normally just the
    CLI/client process. When init_dask_dashboard() creates a
    dask.distributed.Client(), its worker processes are SEPARATE Python
    interpreters (dask's default distributed.worker.multiprocessing-method
    is "spawn" -- confirmed in distributed's own distributed.yaml -- true
    even on Linux, not just Windows/macOS) that never import kazarr.utils
    on their own: the pickled zarr/s3fs write tasks they execute reference
    s3fs/zarr internals, not kazarr, so nothing triggers that import.

    Without this, --s3-upload-stats would silently see nothing for the
    chunk writes that actually happen (which run in the workers) in
    --dask-dashboard mode: "s3://" would resolve to plain
    s3fs.core.S3FileSystem inside every worker process instead of
    _StatsS3FileSystem, so register_stats_handlers() would never run there.

    This is a no-op when Dask is used WITHOUT a distributed.Client() (i.e.
    without --dask-dashboard): dask's default local scheduler then runs
    tasks as threads within the client's own process, which already has
    the registration from kazarr.utils' import, and threads share that
    process's fsspec registry -- this function is simply never called in
    that case.

    dask.distributed is imported lazily here (not at module level) so that
    importing kazarr.s3.upload_stats -- which kazarr.utils does
    unconditionally -- never pulls in dask.distributed for code paths that
    don't use it.
    """
    from distributed import WorkerPlugin

    class _StatsS3SetupPlugin(WorkerPlugin):
        idempotent = True  # one registration is enough per worker

        def setup(self, worker):
            register_stats_s3_filesystem()

    client.register_plugin(_StatsS3SetupPlugin(), name="kazarr-stats-s3fs-setup")
