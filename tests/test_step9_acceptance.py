"""Verify acceptance-report failure handling and CSV comparisons with real fixture models."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import create_app
from app.schemas import EXAMPLE
from src.inference.registry import configure_run
from scripts.step9_verify_project import check_export, save_report, verify_application
from test_step7_api import synthetic_run, RUN_ID


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        synthetic_run(cls.root)
        cls.registry = configure_run(cls.root, RUN_ID, allow_test_artifacts=True)
        source = Path(__file__).resolve().parents[1] / 'examples' / 'step9_carbon_profile.json'
        cls.profile = json.loads(source.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def client(self):
        return TestClient(create_app(self.root, registry=self.registry))

    def test_complete_checks_are_labelled_synthetic_and_reports_are_preserved(self):
        with self.client() as client:
            report = verify_application(client, self.profile)
        self.assertEqual(report['status'], 'synthetic_software_checks_passed')
        self.assertEqual(len(report['checks']), 9)
        self.assertFalse(report['browser_interaction_checked_by_this_script'])
        first = save_report(report, self.root)
        second = save_report(report, self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(first.read_text())['status'], 'synthetic_software_checks_passed')

    def test_unconfigured_models_fail_without_false_success(self):
        with tempfile.TemporaryDirectory() as empty:
            with TestClient(create_app(empty)) as client:
                report = verify_application(client, self.profile)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('503', report['checks'][-1]['error'])

    def test_reordered_batch_is_caught_by_acceptance_check(self):
        original = self.registry.predict
        def wrong_order(requests):
            results = original(requests)
            return list(reversed(results)) if len(requests) == 3 else results
        with patch.object(self.registry, 'predict', side_effect=wrong_order):
            with self.client() as client:
                report = verify_application(client, self.profile)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('order', report['checks'][-1]['error'])

    def export_row(self, prediction):
        p = prediction
        return {'run_id': p['run_id'], 'working_ion': p['working_ion'],
                **{k: str(v) for k, v in p['interval'].items() if k != 'loading_convention'},
                'predicted_voltage_V': str(p['predicted_voltage_V']),
                'model_family': p['model']['family'], 'training_population': p['model']['training_population'],
                'candidate_id': p['model']['candidate_id'], 'evaluation_scope': p['evaluation_reference']['scope'],
                'reference_dataset_mae_V': str(p['evaluation_reference']['all']['mae_V']),
                'extrapolates_working_ion': str(p['extrapolates_working_ion']).lower(),
                'changed_training_constant_features': str(p['changed_training_constant_features']),
                'warnings': ' | '.join(p['warnings'])}

    def write_export(self, row):
        path = self.root / 'export.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        return path

    def test_export_preserves_precision_and_rejects_corruption_or_wrong_run(self):
        with self.client() as client:
            prediction = client.post('/predict', json=EXAMPLE).json()
        row = self.export_row(prediction)
        check_export(self.write_export(row), prediction)
        for field, value in [('predicted_voltage_V', 'NaN'), ('predicted_voltage_V', '900'),
                             ('run_id', 'different'), ('warnings', ''), ('fracA_discharge', '1')]:
            changed = copy.copy(row); changed[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    check_export(self.write_export(changed), prediction)

    def test_missing_optional_export_becomes_reported_failure(self):
        with self.client() as client:
            report = verify_application(client, self.profile, export_csv=self.root / 'missing.csv')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['checks'][-1]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
