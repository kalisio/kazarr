"""JSON serialization with orjson.

orjson serializes numpy arrays natively (no conversion to Python lists), which
is much faster than FastAPI's default serialization for large outputs.
NaN and infinite values are serialized as null.
"""

import numpy as np
import orjson

ORJSON_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS


def _default(obj):
    # Types orjson does not handle natively (or numpy arrays it can't serialize
    # directly: non contiguous, object dtype...)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Type is not JSON serializable: {type(obj).__name__}")


def dumps(obj) -> bytes:
    return orjson.dumps(obj, default=_default, option=ORJSON_OPTIONS)


def json_array(values, flatten=True) -> np.ndarray | list:
    """Convert values into an array orjson can serialize directly.

    float32 (and float16) values are converted to float64 so that they are
    written exactly as with `.tolist()` (e.g. 45.099998474121094 and not 45.1).
    Non numeric arrays are converted to Python lists.
    """
    array = np.asarray(values)
    if flatten:
        array = array.ravel()
    if array.dtype.kind == "f" and array.dtype.itemsize < 8:
        array = array.astype(np.float64)
    elif array.dtype.kind not in "fiub":
        return array.tolist()
    return np.ascontiguousarray(array)
