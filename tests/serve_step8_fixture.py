"""Developer browser-test server using synthetic models; never a research run.

Run with the existing Python environment. The temporary models are deleted on exit.
"""

from pathlib import Path
import sys
import tempfile
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from app.main import create_app
from test_step7_api import synthetic_run, RUN_ID
from src.inference.registry import configure_run
import uvicorn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        synthetic_run(root)
        registry = configure_run(root, RUN_ID, allow_test_artifacts=True)
        print('SYNTHETIC BROWSER VERIFICATION SERVER. NOT A RESEARCH MODEL.', flush=True)
        uvicorn.run(create_app(root, registry=registry), host='127.0.0.1', port=args.port)
