"""Verify a completed training run and configure its saved models for the API."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import create_app
from app.schemas import EXAMPLE
from src.inference.registry import configure_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step6-run', required=True)
    args = parser.parse_args()
    try:
        print('Verifying completed training files and loading all six models...', flush=True)
        registry = configure_run(ROOT, args.step6_run)
        with TestClient(create_app(ROOT, registry=registry)) as client:
            response = client.get('/ready')
            if response.status_code != 200 or response.json()['loaded_models'] != 6:
                raise ValueError('API readiness check failed')
            response = client.post('/predict', json=EXAMPLE)
            if response.status_code != 200:
                raise ValueError('API prediction check failed: ' + response.text)
            result = response.json()
            response = client.post('/predict', json={**EXAMPLE, 'formula_discharge': 'NaFePO4'})
            if response.status_code != 422:
                raise ValueError('API did not reject an inconsistent reaction')
        print('[OK] All six bundles loaded and produced finite predictions')
        print('[OK] Readiness, prediction and invalid-input checks passed')
        print(f'Active run: {registry.run_id}')
        print('Default population routing: Li/Na/K -> li_only; Mg/Ca/Zn/Al/Y -> mixed_ions')
        for population, family in registry.defaults.items():
            print(f'CV-selected family for {population}: {family}')
        print(f"Example FePO4 -> LiFePO4 prediction: {result['predicted_voltage_V']:.4f} V")
        print('The example uses the supplied symmetry description; it is not an independent accuracy test.')
        print('STEP 7 API CONFIGURATION AND LOCAL CHECK PASSED')
        print('Next: start Uvicorn and open http://127.0.0.1:8000/docs in Chrome.')
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'[ERROR] {exc}')
        print('Share this error. Training files have not been modified or retrained.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
