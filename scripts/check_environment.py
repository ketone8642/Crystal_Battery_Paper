r"""Check Step 2 dependencies without downloading data or training a model.

Run from the project root with: python scripts\check_environment.py
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-dotenv": "dotenv",
    "pytest": "pytest",
    "httpx": "httpx",
}


def main() -> int:
    failures = 0
    print(f"Python: {platform.python_version()} ({struct.calcsize('P') * 8}-bit)")
    print(f"Executable: {sys.executable}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    if sys.version_info[:2] != (3, 11):
        print("[FAIL] This setup targets Python 3.11. Create the environment using py -3.11 -m venv .venv")
        failures += 1
    if struct.calcsize("P") != 8:
        print("[FAIL] Install a 64-bit Python interpreter.")
        failures += 1
    if sys.prefix == sys.base_prefix:
        print("[FAIL] A virtual environment is not active. Run .venv\\Scripts\\activate.bat in Command Prompt.")
        failures += 1

    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f"[OK] {distribution} {version(distribution)}")
        except Exception as exc:
            print(f"[FAIL] {distribution}: {type(exc).__name__}: {exc}")
            failures += 1

    # A subprocess isolates native TensorFlow DLL/import failures and provides
    # a bounded arithmetic check. This does not train a network.
    tf_code = (
        "import tensorflow as tf; "
        "value = float(tf.reduce_sum(tf.constant([1.0, 2.0, 3.0])).numpy()); "
        "assert value == 6.0, value; "
        "print('[OK] TensorFlow ' + tf.__version__ + ' CPU calculation = 6.0')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", tf_code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            print(f"[FAIL] TensorFlow check exited with code {result.returncode}.")
            print((result.stderr or result.stdout)[-3500:])
            failures += 1
        else:
            print(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print("[FAIL] TensorFlow import/calculation exceeded 120 seconds. Try running it again and report the error if it persists.")
        failures += 1

    try:
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/health")
            body = response.json()
            if (response.status_code != 200 or body.get('status') != 'ok'
                    or not isinstance(body.get('model_loaded'), bool)
                    or body.get('stage') not in ('environment_setup', 'prediction_api')):
                raise RuntimeError("Unexpected health response")
            home = client.get("/")
            if home.status_code != 200 or "Battery Electrode Voltage Prediction" not in home.text:
                raise RuntimeError("Setup HTML page is unavailable")
        print("[OK] FastAPI health response and setup page")
    except Exception as exc:
        print(f"[FAIL] FastAPI check: {type(exc).__name__}: {exc}")
        failures += 1

    print()
    if failures:
        print(f"STEP 2 NEEDS ATTENTION: {failures} check(s) failed.")
        print("Share the [FAIL] lines and the final installation error, if any.")
        return 1
    print("STEP 2 ENVIRONMENT CHECK PASSED")
    print("Next: start the server and verify the page in Chrome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
