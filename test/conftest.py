"""
conftest.py — Shared pytest configuration and fixtures.

This file is automatically loaded by pytest before any test file.
It handles sys.path manipulation for both the main project and the
conversion_tool subproject, eliminating the need to do so in each test file.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.utils.file import load
from src.api.api import app

_root_dir = Path(__file__).resolve().parent.parent
_tool_dir = _root_dir / "conversion_tool"

# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------
TEST_TMP_FOLDER = "test/tests_tmp"
os.environ.setdefault("TEST_TMP_FOLDER", TEST_TMP_FOLDER)
os.makedirs(TEST_TMP_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Global fixtures (run for every test)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the dataset loading cache before each test to prevent interference."""
    load.cache_clear()

# ---------------------------------------------------------------------------
# Session-scoped fixtures (created once for the whole test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI test client shared across all tests."""
    return TestClient(app)


@pytest.fixture(scope="session")
def convert():
    """
    Execute the conversion_tool in an isolated subprocess.
    Returns standard output (stdout) on success, raises an error on failure.
    """
    def _run(
        dataset_name: str,
        input_path: str,
        template=None,
        config=None,
        config_file=None,
        description=None,
        output_path=None,
        pipeline_name=None,
        templates_path=None,
        data_mapping=None,
        mesh_type=None,
        dask_dashboard=False,
    ) -> str:
        # This function use subprocess to run the conversion tool 
        # to avoid conflicts with the main app's imports and environment.

        entry_script = _tool_dir / "main.py"
        tool_command = "new-dataset"

        cmd = [sys.executable, str(entry_script), tool_command]
        opt_args = {
            "template": template,
            "config": json.dumps(config) if config is not None else None,
            "config_file": config_file,
            "description": description,
            "output": output_path,
            "pipeline": pipeline_name,
            "templates_path": templates_path,
            "data_mapping": data_mapping,
            "mesh_type": mesh_type,
            "dask_dashboard": dask_dashboard,
        }
        for arg_name, arg_value in opt_args.items():
            if isinstance(arg_value, bool):
                if arg_value:
                    cmd.append(f"--{arg_name.replace('_', '-')}")
            elif arg_value is not None:
                cmd += [f"--{arg_name.replace('_', '-')}", str(arg_value)]

        cmd += [dataset_name, input_path]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(_tool_dir)

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        print(f"\n[KAZARR - Conversion tool] \n {result.stdout}")

        if result.returncode != 0:
            raise RuntimeError(
                f"Conversion tool failed (Code {result.returncode})\n"
                f"STDERR:\n{result.stderr}\n"
                f"STDOUT:\n{result.stdout}"
            )

        return result.stdout

    return _run
