"""API integration using actual models trained on isolated synthetic test data."""

import csv
import io
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import EXAMPLE
from src.features.preprocessing import FeatureData, fit_preprocessor
from src.features.reaction_features import FEATURE_NAMES, feature_matrix
from src.inference.registry import CONFIG_RELATIVE, ModelRegistry, configure_run
from src.training.experiment import make_bundle
from src.training.io_utils import read_json, write_json, sha
from src.training.voltage_models import candidates, runtime_versions, metrics


RUN_ID = 'step6_000000000007'
FAMILIES = ('svr', 'krr', 'dnn')
POPULATIONS = ('mixed_ions', 'li_only')


def synthetic_run(root):
    """Serving fixture only; synthetic metrics are never research results."""
    config = {'purpose': 'synthetic_smoke', 'epochs': 2, 'seed': 31, 'batch_size': 8}
    versions = runtime_versions()
    report_dir = root / 'reports' / RUN_ID
    report_dir.mkdir(parents=True)
    report = {'run_id': RUN_ID, 'configuration': config, 'populations': {}}
    selected = {}
    for population in POPULATIONS:
        rng = np.random.default_rng(83)
        X = np.zeros((24, len(FEATURE_NAMES)))
        X[:, :12] = rng.normal(size=(24, 12))
        y = 2.0 + X[:, 0] * 0.2
        ions = np.full(24, 'Li', dtype='U2')
        if population == 'mixed_ions':
            ions = np.array(['Li', 'Mg', 'Ca', 'Zn', 'Al', 'Y'] * 4)
        development = FeatureData(X, y, np.array([f'r{i}' for i in range(24)]),
            np.array([f'g{i//2}' for i in range(24)]), ions, np.array([(i//2) % 2 for i in range(24)]),
            'development', population)
        pp_path = root / (population + '_pp.npz')
        fit_preprocessor(development, n_components=2).save(pp_path)
        info = {'bundles': {}, 'selected': {}, 'evaluation': {}, 'cv_selected_family': 'dnn'}
        selected[population] = {}
        for family in FAMILIES:
            candidate = next(c for c in candidates() if c['id'] == family + '_scaled_0')
            chosen = {'candidate': candidate, 'mean_fold_mae_V': 0.5}  # Fixture placeholder, no real CV claim.
            directory = root / 'models' / population / RUN_ID / family
            bundle = make_bundle(development, pp_path, chosen, directory, config, versions)
            scores = metrics(y, bundle.predict_matrix(X))
            info['bundles'][family] = directory.relative_to(root).as_posix()
            info['selected'][family] = chosen
            selected[population][family] = candidate
            info['evaluation'][family] = {
                'holdout': {'all': scores, 'per_ion': {str(ion): scores for ion in set(ions)}},
                'held_out_Na': {'all': scores, 'known_development_group': scores, 'new_development_group': scores},
                'held_out_K': {'all': scores, 'known_development_group': scores, 'new_development_group': scores}}
        report['populations'][population] = info
    selection = {'selection_uses_holdout_or_transfer': False, 'selected': selected,
                 'cv_selected_family': {p: 'dnn' for p in POPULATIONS}}
    write_json(report_dir / 'selection.json', selection)
    report['selection_sha256_before_evaluation'] = sha(report_dir / 'selection.json')
    write_json(report_dir / 'step6_report.json', report)
    files = [p for p in (root / 'models').rglob('*') if p.is_file()] + list(report_dir.glob('*.json'))
    write_json(report_dir / 'step6_manifest.json', {'status': 'complete', 'configuration': config,
        'output_sha256': {p.relative_to(root).as_posix(): sha(p) for p in files}})
    source = Path(__file__).resolve().parents[1] / 'frontend'
    shutil.copytree(source, root / 'frontend')


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        synthetic_run(cls.root)
        cls.registry = configure_run(cls.root, RUN_ID, allow_test_artifacts=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.client = TestClient(create_app(self.root, registry=self.registry))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_readiness_docs_and_status_page(self):
        self.assertEqual(self.client.get('/ready').json()['loaded_models'], 6)
        self.assertTrue(self.client.get('/health').json()['model_loaded'])
        self.assertEqual(self.client.get('/docs').status_code, 200)
        self.assertIn('/predict/profile', self.client.get('/openapi.json').json()['paths'])
        self.assertIn('Battery Electrode Voltage Prediction', self.client.get('/').text)

    def test_prediction_equals_direct_saved_bundle(self):
        result = self.client.post('/predict', json=EXAMPLE).json()
        expected = self.registry.bundles[('li_only', 'dnn')].predict_matrix(feature_matrix([EXAMPLE]))[0]
        self.assertAlmostEqual(result['predicted_voltage_V'], expected, places=6)
        self.assertAlmostEqual(result['interval']['fracA_discharge'], 1/7)
        self.assertEqual(result['interval']['ion_per_host_discharge'], 1)

    def test_all_six_models_are_selectable(self):
        for population in POPULATIONS:
            for family in FAMILIES:
                response = self.client.post('/predict', json={**EXAMPLE, 'model_family': family, 'training_population': population})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result['model']['family'], family)
                self.assertEqual(result['model']['training_population'], population)

    def test_default_ion_routing_and_transfer_notices(self):
        for ion in ('Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Al', 'Y'):
            response = self.client.post('/predict', json={**EXAMPLE, 'working_ion': ion, 'formula_discharge': ion + 'FePO4'})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result['model']['training_population'], 'li_only' if ion in ('Li', 'Na', 'K') else 'mixed_ions')
            self.assertEqual(result['extrapolates_working_ion'], ion in ('Na', 'K'))
            if ion in ('Na', 'K'):
                self.assertIn('new_development_group', result['evaluation_reference'])
                self.assertTrue(any('absent' in w for w in result['warnings']))

    def test_explicit_mixed_sodium_route_is_identified(self):
        body = {**EXAMPLE, 'working_ion': 'Na', 'formula_discharge': 'NaFePO4', 'training_population': 'mixed_ions'}
        result = self.client.post('/predict', json=body).json()
        self.assertEqual(result['model']['training_population'], 'mixed_ions')
        self.assertTrue(result['extrapolates_working_ion'])

    def test_li_only_multivalent_route_is_rejected(self):
        body = {**EXAMPLE, 'working_ion': 'Mg', 'formula_discharge': 'MgFePO4', 'training_population': 'li_only'}
        self.assertEqual(self.client.post('/predict', json=body).status_code, 422)

    def test_target_and_provenance_are_not_accepted_as_predictors(self):
        for key in ('average_voltage_V', 'row_uid', 'group_id', 'warnings'):
            self.assertEqual(self.client.post('/predict', json={**EXAMPLE, key: 'unwanted'}).status_code, 422)

    def test_bad_host_loading_symmetry_and_formula_are_rejected(self):
        for changes in ({'formula_discharge': 'LiCoO2'}, {'formula_discharge': 'FePO4'},
                        {'spacegroup_number_charge': 225}, {'formula_charge': '=1+2'},
                        {'formula_charge': 'Fe' * 300}, {'fracA_discharge': 0.5}):
            self.assertEqual(self.client.post('/predict', json={**EXAMPLE, **changes}).status_code, 422)

    def test_strict_numbers_and_nonfinite_json_have_safe_errors(self):
        for changes in ({'spacegroup_number_charge': True}, {'spacegroup_number_charge': '62'},
                        {'fracA_discharge': True}, {'fracA_discharge': '0.142857'}):
            self.assertEqual(self.client.post('/predict', json={**EXAMPLE, **changes}).status_code, 422)
        import json
        raw = json.dumps({**EXAMPLE, 'fracA_discharge': float('nan')})
        response = self.client.post('/predict', content=raw, headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 422)
        self.assertIn('detail', response.json())

    def test_batch_preserves_order_and_matches_individual_predictions(self):
        items = [{**EXAMPLE, 'model_family': f} for f in ('svr', 'dnn', 'krr', 'svr')]
        response = self.client.post('/predict/batch', json={'items': items})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['count'], 4)
        for body, prediction in zip(items, result['predictions']):
            separate = self.client.post('/predict', json=body).json()
            self.assertAlmostEqual(prediction['predicted_voltage_V'], separate['predicted_voltage_V'], places=6)

    def test_invalid_batch_is_rejected_before_model_execution(self):
        with patch.object(self.registry, 'predict', side_effect=AssertionError('must not predict')):
            response = self.client.post('/predict/batch', json={'items': [EXAMPLE, {**EXAMPLE, 'formula_discharge': 'CoO2'}]})
            self.assertEqual(response.status_code, 422)

    def test_batch_size_bounds(self):
        for items in ([], [EXAMPLE] * 129):
            self.assertEqual(self.client.post('/predict/batch', json={'items': items}).status_code, 422)

    def test_csv_contains_the_same_predictions(self):
        response = self.client.post('/predict/batch.csv', json={'items': [EXAMPLE]})
        self.assertEqual(response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(response.text)))
        expected = self.client.post('/predict', json=EXAMPLE).json()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]['predicted_voltage_V']), expected['predicted_voltage_V'], places=6)
        self.assertIn('attachment', response.headers['content-disposition'])

    def test_profile_returns_interval_loadings_without_interpolation(self):
        # Deliberately synthetic stoichiometric continuation; no stability claim.
        second = {**EXAMPLE, 'formula_charge': 'LiFePO4', 'formula_discharge': 'Li2FePO4'}
        response = self.client.post('/predict/profile', json={'intervals': [EXAMPLE, second]})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['predictions'][0]['interval']['ion_per_host_discharge'], 1)
        self.assertEqual(result['predictions'][1]['interval']['ion_per_host_charge'], 1)
        self.assertEqual(result['predictions'][1]['interval']['ion_per_host_discharge'], 2)

    def test_profile_rejects_gaps_overlaps_and_inconsistent_shared_endpoints(self):
        second = {**EXAMPLE, 'formula_charge': 'LiFePO4', 'formula_discharge': 'Li2FePO4'}
        bad = [{**second, 'formula_charge': 'Li2FePO4', 'formula_discharge': 'Li3FePO4'},
               EXAMPLE, {**second, 'spacegroup_number_charge': 63},
               {**second, 'model_family': 'svr'}]
        for item in bad:
            self.assertEqual(self.client.post('/predict/profile', json={'intervals': [EXAMPLE, item]}).status_code, 422)

    def test_model_catalog_uses_saved_evaluation(self):
        result = self.client.get('/models').json()
        self.assertEqual(len(result['models']), 6)
        card = next(c for c in result['models'] if c['family'] == 'dnn' and c['training_population'] == 'li_only')
        self.assertEqual(card['holdout'], self.registry.report['populations']['li_only']['evaluation']['dnn']['holdout'])
        self.assertIn('not confidence intervals', result['evaluation_note'])

    def test_unconfigured_app_is_live_but_not_ready(self):
        with tempfile.TemporaryDirectory() as temp, TestClient(create_app(temp)) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertFalse(client.get('/health').json()['model_loaded'])
            self.assertEqual(client.get('/ready').status_code, 503)
            self.assertEqual(client.post('/predict', json=EXAMPLE).status_code, 503)

    def test_production_loader_rejects_synthetic_test_models(self):
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            ModelRegistry(self.root, RUN_ID)

    def test_configured_lifespan_loads_models(self):
        with TestClient(create_app(self.root, allow_test_artifacts=True)) as client:
            self.assertEqual(client.get('/ready').json()['loaded_models'], 6)
            self.assertEqual(client.post('/predict', json=EXAMPLE).status_code, 200)

    def test_tampered_bundle_is_rejected(self):
        path = self.root / 'models' / 'li_only' / RUN_ID / 'dnn' / 'regressor.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                ModelRegistry(self.root, RUN_ID, allow_test_artifacts=True)
        finally:
            path.write_bytes(original)

    def test_failed_configuration_preserves_the_previous_active_run(self):
        path = self.root / 'reports' / RUN_ID / 'step6_manifest.json'
        original = path.read_bytes()
        old_config = (self.root / CONFIG_RELATIVE).read_bytes()
        try:
            value = read_json(path)
            value['status'] = 'failed'
            write_json(path, value)
            with self.assertRaises(ValueError):
                configure_run(self.root, RUN_ID, allow_test_artifacts=True)
            self.assertEqual((self.root / CONFIG_RELATIVE).read_bytes(), old_config)
        finally:
            path.write_bytes(original)

    def test_model_error_returns_no_substitute_prediction(self):
        with patch.object(self.registry, 'predict', side_effect=ValueError('test inference failure')):
            with self.assertLogs('battery_voltage.api', level='ERROR'):
                response = self.client.post('/predict', json=EXAMPLE)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('predicted_voltage_V', response.json())


if __name__ == '__main__':
    unittest.main()
