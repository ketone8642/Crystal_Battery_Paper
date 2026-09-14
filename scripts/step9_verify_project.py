"""Verify the installed application with saved models; never fit or tune models."""

from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def check_export(path, prediction):
    """Compare a single FePO4 browser export with its current API prediction."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, 'Use the single FePO4-example export, containing one data row.')
    row = rows[0]
    expected_text = {
        'run_id': prediction['run_id'], 'working_ion': prediction['working_ion'],
        'formula_charge': prediction['interval']['formula_charge'],
        'formula_discharge': prediction['interval']['formula_discharge'],
        'model_family': prediction['model']['family'],
        'training_population': prediction['model']['training_population'],
        'candidate_id': prediction['model']['candidate_id'],
        'evaluation_scope': prediction['evaluation_reference']['scope'],
    }
    for name, expected in expected_text.items():
        require(row.get(name) == expected, f'Export mismatch in {name}; use this run and the default FePO4 example.')
    for name in ('ion_per_host_charge', 'ion_per_host_discharge', 'fracA_charge', 'fracA_discharge'):
        require(near(row[name], prediction['interval'][name]), f'Export mismatch in {name}.')
    require(near(row['predicted_voltage_V'], prediction['predicted_voltage_V']), 'Export voltage does not match the active model.')
    require(near(row['reference_dataset_mae_V'], prediction['evaluation_reference']['all']['mae_V']), 'Export reference MAE does not match.')
    require(row.get('extrapolates_working_ion', '').lower() == str(prediction['extrapolates_working_ion']).lower(), 'Export transfer flag does not match.')
    require(row.get('changed_training_constant_features') == str(prediction['changed_training_constant_features']), 'Export descriptor notice count does not match.')
    require(row.get('warnings') == ' | '.join(prediction['warnings']), 'Export warnings do not match.')
    return {'filename': Path(path).name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'numeric_comparison_tolerance': 'relative=1e-6, absolute=1e-6',
            'limitation': 'The browser CSV omits space groups; this check is specifically for the documented default FePO4 example.'}


def verify_application(client, profile_body, *, export_csv=None):
    from app.schemas import EXAMPLE
    from scripts.step8_check_ui import AssetLinks

    report = {'status': 'failed', 'checks': [], 'model_data_source': None,
              'scope': 'Application acceptance checks using existing saved models; no fitting, tuning or accuracy reevaluation.',
              'browser_interaction_checked_by_this_script': False,
              'pending_manual_checks': ['Chrome batch upload/download', 'Chrome carbon-profile chart', 'Chrome model-results tab']}

    def step(name, operation):
        value = operation()
        report['checks'].append({'name': name, 'status': 'passed'})
        print('[OK] ' + name, flush=True)
        return value

    def response(method, path, body=None, status=200):
        r = client.get(path) if method == 'GET' else client.post(path, json=body)
        require(r.status_code == status, f'{path} returned HTTP {r.status_code}; expected {status}: {r.text[:350]}')
        return r

    def predict(body):
        p = response('POST', '/predict', body).json()
        require(math.isfinite(p['predicted_voltage_V']), 'A prediction is nonfinite.')
        require(p['run_id'] == report['run_id'], 'A prediction uses a different run.')
        return p

    try:
        def ready():
            health = response('GET', '/health').json()
            loaded = response('GET', '/ready').json()
            require(health['model_loaded'] and health['loaded_models'] == loaded['loaded_models'] == 6, 'All six models must be ready.')
            require(health['run_id'] == loaded['run_id'], 'Health and readiness identify different runs.')
            return loaded['run_id']
        report['run_id'] = step('Six saved models are ready', ready)

        def catalog_check():
            catalog = response('GET', '/models').json()
            expected = {(p, f) for p in ('li_only', 'mixed_ions') for f in ('dnn', 'svr', 'krr')}
            require(catalog['run_id'] == report['run_id'], 'Catalog run does not match readiness.')
            require(len(catalog['models']) == 6 and {(m['training_population'], m['family']) for m in catalog['models']} == expected, 'Catalog must identify six distinct model choices.')
            for population, family in catalog['default_family_by_population'].items():
                chosen = [m for m in catalog['models'] if m['training_population'] == population and m['selected_by_cv']]
                require(len(chosen) == 1 and chosen[0]['family'] == family, 'CV selection badge differs from routing.')
            return catalog
        catalog = step('Model catalog and CV selections agree', catalog_check)
        report['model_data_source'] = catalog['data_source']
        report['saved_evaluation_catalog'] = catalog

        def assets():
            page = response('GET', '/')
            parser = AssetLinks(); parser.feed(page.text)
            require({'single-form', 'batch-form', 'profile-form', 'models-table'} <= parser.ids, 'Step 8 page is missing a workspace section.')
            for url in parser.assets + ['/static/ui_utils.mjs']:
                asset = response('GET', url)
                if any(ext in url for ext in ('.js', '.mjs')):
                    require('javascript' in asset.headers.get('content-type', ''), 'Incorrect module content type: ' + url)
            paths = response('GET', '/openapi.json').json()['paths']
            require({'/predict', '/predict/batch', '/predict/profile', '/predict/batch.csv'} <= set(paths), 'Expected prediction route is missing.')
        step('Four browser sections, module assets and API schema are available', assets)

        def single():
            p = predict(EXAMPLE)
            require(p['model']['training_population'] == 'li_only' and p['model']['family'] == catalog['default_family_by_population']['li_only'], 'Automatic Li routing differs from its CV selection.')
            require(near(p['interval']['fracA_discharge'], 1/7) and near(p['interval']['ion_per_host_discharge'], 1), 'FePO4 concentration conventions are inconsistent.')
            return p
        example = step('Default FePO4 prediction has correct routing and concentrations', single)
        report['fe_po4_example'] = example

        def all_models():
            values = []
            for card in catalog['models']:
                body = {**EXAMPLE, 'model_family': card['family'], 'training_population': card['training_population']}
                p = predict(body)
                for field in ('family', 'training_population', 'candidate_id', 'representation', 'training_rows'):
                    require(p['model'][field] == card[field], 'Selected model does not match catalog: ' + field)
                values.append({'population': card['training_population'], 'family': card['family'], 'voltage_V': p['predicted_voltage_V']})
            return values
        report['six_model_inference_examples'] = step('All six saved model choices produce finite predictions', all_models)

        batch_items = [{**EXAMPLE, 'model_family': family} for family in ('dnn', 'svr', 'krr')]
        def batch():
            data = response('POST', '/predict/batch', {'items': batch_items}).json()
            require(data['count'] == len(data['predictions']) == 3, 'Batch row count differs.')
            for body, p in zip(batch_items, data['predictions']):
                require(p['model']['family'] == body['model_family'], 'Batch changed request order.')
                require(near(p['predicted_voltage_V'], predict(body)['predicted_voltage_V']), 'Batch and individual inference differ.')
            return data
        batch_result = step('Batch results preserve order and agree with individual inference', batch)

        def csv_check():
            r = response('POST', '/predict/batch.csv', {'items': batch_items})
            require('text/csv' in r.headers.get('content-type', '') and 'attachment' in r.headers.get('content-disposition', ''), 'CSV download headers are missing.')
            rows = list(csv.DictReader(io.StringIO(r.text)))
            require(len(rows) == 3, 'CSV row count differs.')
            for row, p in zip(rows, batch_result['predictions']):
                require(row['run_id'] == p['run_id'] and row['model_family'] == p['model']['family'], 'CSV provenance/order differs.')
                require(near(row['predicted_voltage_V'], p['predicted_voltage_V']), 'CSV prediction differs from JSON.')
        step('API CSV download agrees with JSON predictions', csv_check)

        def profile():
            data = response('POST', '/predict/profile', profile_body).json()
            require(data['count'] == len(data['predictions']) == 2, 'Carbon example must produce two intervals.')
            for body, p in zip(profile_body['intervals'], data['predictions']):
                require(near(p['predicted_voltage_V'], predict(body)['predicted_voltage_V']), 'Profile and individual predictions differ.')
            bounds = [(p['interval']['ion_per_host_charge'], p['interval']['ion_per_host_discharge']) for p in data['predictions']]
            require(all(near(a, b) for pair, expected in zip(bounds, [(0, 1/12), (1/12, 1/6)]) for a, b in zip(pair, expected)), 'Carbon loading axis uses an incorrect host convention.')
            return data
        report['carbon_profile_example'] = step('Dataset carbon profile returns two contiguous interval averages', profile)

        def invalid():
            for body in ({**EXAMPLE, 'formula_discharge': 'LiCoO2'}, {**EXAMPLE, 'spacegroup_number_charge': 225}, {**EXAMPLE, 'fracA_discharge': 0.5}):
                response('POST', '/predict', body, status=422)
            response('POST', '/predict/batch', {'items': []}, status=422)
            response('POST', '/predict/profile', {'intervals': list(reversed(profile_body['intervals']))}, status=422)
        step('Invalid host, symmetry, concentration, batch and profile inputs are rejected', invalid)

        if export_csv is not None:
            report['browser_export'] = step('Uploaded single-prediction CSV agrees with the active example', lambda: check_export(export_csv, example))
        else:
            report['browser_export'] = {'status': 'not_requested', 'command_option': '--export-csv PATH'}

        if 'synthetic' in catalog['data_source'].lower():
            report['status'] = 'synthetic_software_checks_passed'
            report['limitation'] = 'These fixture-model outputs cannot complete research-model acceptance.'
        else:
            report['status'] = 'automated_checks_passed'
    except Exception as exc:
        report['checks'].append({'name': 'Verification stopped', 'status': 'failed', 'error': str(exc)})
        print('[ERROR] ' + str(exc), flush=True)
    return report


def save_report(report, root):
    directory = Path(root) / 'reports' / ('step9_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'step9_report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Final application verification', '', 'Status: ' + report['status'], '',
             'Active run: ' + report.get('run_id', 'unavailable'), '', report['scope'], '']
    lines.extend('- ' + item['status'].upper() + ': ' + item['name'] + (': ' + item['error'] if 'error' in item else '') for item in report['checks'])
    lines.extend(['', 'Browser actions still require manual confirmation:', ''])
    lines.extend('- ' + item for item in report['pending_manual_checks'])
    if 'fe_po4_example' in report:
        lines.extend(['', 'FePO4 example voltage: ' + str(report['fe_po4_example']['predicted_voltage_V']) + ' V.'])
    (directory / 'step9_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-csv', type=Path, help='Optional browser CSV from the default single FePO4 example.')
    args = parser.parse_args()
    from fastapi.testclient import TestClient
    from app.main import create_app
    profile_path = ROOT / 'examples' / 'step9_carbon_profile.json'
    try:
        profile_body = json.loads(profile_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        print('[ERROR] Install examples/step9_carbon_profile.json from the Step 9 addon: ' + str(exc))
        return 1
    with TestClient(create_app(ROOT)) as client:
        report = verify_application(client, profile_body, export_csv=args.export_csv)
    report['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                         'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'fastapi', 'httpx')}}
    report['profile_input_sha256'] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    path = save_report(report, ROOT)
    print('Report: ' + str(path))
    passed = report['status'] == 'automated_checks_passed'
    print('STEP 9 AUTOMATED APPLICATION CHECK PASSED' if passed else 'STEP 9 CHECK DID NOT COMPLETE FOR RESEARCH MODELS')
    print('Complete the three Chrome checks in README_COMPLETE.md; no model was trained.')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
