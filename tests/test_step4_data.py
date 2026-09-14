"""Synthetic software regressions, separate from the user's research data."""

from copy import deepcopy
import csv
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data.chemistry import composition, host_and_loading
from src.data.preparation import Settings, group_rows, prepare, stable_hash, validate_rows

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('step4_script', ROOT / 'scripts' / 'step4_prepare_data.py')
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def example(n=1, ion='Li'):
    host = f'Fe{n}O{2*n+1}'
    return {
        'record_key': stable_hash(['synthetic', n, ion]), 'battery_id': '',
        'row_type': 'adjacent', 'interval_index': '0', 'working_ion': ion,
        'framework_formula': host, 'formula_charge': host, 'formula_discharge': ion + host,
        'id_charge': f'mp-host-{n}', 'id_discharge': f'mp-{ion}-{n}',
        'fracA_charge': '0', 'fracA_discharge': str(1/(3*n+2)), 'average_voltage_V': '3.4',
        'crystal_system_charge': 'Orthorhombic', 'crystal_system_discharge': 'Orthorhombic',
        'spacegroup_number_charge': '62', 'spacegroup_number_discharge': '62',
        'issues': '', 'warnings': '',
    }


def dataset():
    return [example(n, ion) for n in range(1, 41) for ion in ('Li', 'Mg')] + [
        example(n, ion) for n in range(35, 46) for ion in ('Na', 'K')
    ]


class ChemistryTests(unittest.TestCase):
    def test_nested_groups_and_repeated_elements(self):
        self.assertEqual(dict(composition('K4[Fe(CN)6]')), {'K': 4, 'Fe': 1, 'C': 6, 'N': 6})
        self.assertEqual(dict(composition('CH3COOH')), {'C': 2, 'H': 4, 'O': 2})
        self.assertEqual(dict(composition('Ca3(PO4)2')), {'Ca': 3, 'P': 2, 'O': 8})

    def test_invalid_and_unsupported_notation_is_rejected(self):
        for formula in ('', 'Xx2O', 'Li0O', 'Li0.5FePO4', 'CuSO4·5H2O', 'Fe(OH', 'Fe()2',
                        'Fe[OH)2', '2H2O', 'SO4-2', 'Fe 2 O3'):
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                composition(formula)

    def test_normalization_preserves_host_and_loading_across_formula_multiples(self):
        host, loading, fraction = host_and_loading('Li2Fe2(PO4)2', 'Li')
        host2, loading2, fraction2 = host_and_loading('LiFePO4', 'Li')
        self.assertEqual(host, host2)
        self.assertEqual((loading, loading2), (1, 1))
        self.assertAlmostEqual(fraction, 1/7)
        self.assertEqual(fraction, fraction2)

    def test_pure_working_ion_has_no_host(self):
        with self.assertRaisesRegex(ValueError, 'no non-working-ion host'):
            host_and_loading('Li', 'Li')


class ValidationTests(unittest.TestCase):
    def validate(self, records):
        return validate_rows(records, Settings())

    def test_fraction_mismatch_is_rejected(self):
        raw = example()
        raw['fracA_discharge'] = '0.9'
        rows, _ = self.validate([raw])
        self.assertFalse(rows[0]['eligible_for_baseline'])
        self.assertIn('formula_fraction_mismatch_discharge', rows[0]['step4_issues'])

    def test_framework_mismatch_is_rejected(self):
        raw = example()
        raw['framework_formula'] = 'CoO3'
        rows, _ = self.validate([raw])
        self.assertIn('framework_composition_mismatch', rows[0]['step4_issues'])

    def test_missing_symmetry_is_quarantined_without_label_change(self):
        raw = example()
        raw['spacegroup_number_charge'] = ''
        rows, _ = self.validate([raw])
        self.assertFalse(rows[0]['eligible_for_baseline'])
        self.assertTrue(rows[0]['composition_checks_passed'])
        self.assertEqual(rows[0]['average_voltage_V'], raw['average_voltage_V'])

    def test_spacegroup_system_disagreement_is_rejected(self):
        raw = example()
        raw['spacegroup_number_charge'] = '225'
        rows, _ = self.validate([raw])
        self.assertIn('spacegroup_crystal_system_mismatch_charge', rows[0]['step4_issues'])

    def test_negative_zero_and_high_finite_voltages_are_retained(self):
        for voltage in ('-7.75', '0.0', '33.06577115833334'):
            raw = example()
            raw['average_voltage_V'] = voltage
            rows, _ = self.validate([raw])
            self.assertTrue(rows[0]['eligible_for_baseline'])
            self.assertEqual(rows[0]['average_voltage_V'], voltage)

    def test_nonfinite_voltage_is_rejected(self):
        for voltage in ('NaN', 'inf', ''):
            raw = example()
            raw['average_voltage_V'] = voltage
            rows, _ = self.validate([raw])
            self.assertIn('invalid_voltage_label', rows[0]['step4_issues'])

    def test_shared_endpoint_conflict_flags_all_affected_rows(self):
        a, b = example(), example(2)
        b['id_charge'] = a['id_charge']
        rows, report = self.validate([a, b])
        self.assertEqual(len(report['conflicting_endpoint_metadata_ids']), 1)
        self.assertTrue(all('conflicting_endpoint_metadata' in row['step4_issues'] for row in rows))

    def test_exact_duplicate_is_collapsed_but_conflicting_labels_are_not_averaged(self):
        a = example()
        rows, report = self.validate([a, deepcopy(a)])
        self.assertEqual(report['exact_duplicate_rows_excluded'], 1)
        self.assertEqual(sum(r['eligible_for_baseline'] for r in rows), 1)
        b = dict(a, record_key=stable_hash('other-source'), average_voltage_V='4.5')
        rows, report = self.validate([a, b])
        self.assertEqual(report['conflicting_endpoint_reaction_label_groups'], 1)
        self.assertFalse(any(r['eligible_for_baseline'] for r in rows))
        self.assertEqual([r['average_voltage_V'] for r in rows], ['3.4', '4.5'])


class SplitTests(unittest.TestCase):
    def test_transitive_connections_are_preserved(self):
        rows = [
            dict(row_uid='a', record_key='r1', canonical_host='h1', id_charge='m1', id_discharge='m2'),
            dict(row_uid='b', record_key='r2', canonical_host='h2', id_charge='m2', id_discharge='m3'),
            dict(row_uid='c', record_key='r2', canonical_host='h3', id_charge='m4', id_discharge='m5'),
            dict(row_uid='d', record_key='r4', canonical_host='h3', id_charge='m6', id_discharge='m7'),
        ]
        info = group_rows(rows)
        self.assertEqual(info['connected_groups_all_rows'], 1)
        self.assertEqual(len({r['group_id'] for r in rows}), 1)

    def test_splits_do_not_depend_on_targets_or_input_order(self):
        raw = dataset()
        baseline, _ = prepare(raw)
        changed = [dict(r, average_voltage_V=str(i - 40)) for i, r in enumerate(raw)]
        altered, _ = prepare(list(reversed(changed)))
        def assignments(rows):
            return {r['row_uid']: (r['partition'], r['mixed_cv_fold'], r['li_cv_fold']) for r in rows}
        self.assertEqual(assignments(baseline), assignments(altered))

    def test_endpoint_host_and_record_separation_and_fold_coverage(self):
        rows, report = prepare(dataset())
        for model, ions, field in [('mixed_ions', ('Li', 'Mg'), 'mixed_cv_fold'), ('li_only', ('Li',), 'li_cv_fold')]:
            dev = [r for r in rows if r['partition'] == 'development' and r['working_ion'] in ions]
            held = [r for r in rows if r['partition'] == 'holdout' and r['working_ion'] in ions]
            self.assertFalse({r[k] for r in dev for k in ['id_charge', 'id_discharge']} &
                             {r[k] for r in held for k in ['id_charge', 'id_discharge']})
            self.assertEqual(set(r[field] for r in dev), set(range(10)))
            self.assertTrue(all(r[field] is None for r in held))
            self.assertEqual(sum(f['validation_rows'] for f in report['split_checks'][model]['cross_validation']), len(dev))
        self.assertTrue(all(r['partition'] == 'held_out_' + r['working_ion'] for r in rows if r['working_ion'] in ('Na', 'K')))

    def test_insufficient_groups_fail_instead_of_falling_back_to_row_split(self):
        with self.assertRaisesRegex(ValueError, 'groups'):
            prepare([example()])


class ExportTests(unittest.TestCase):
    def test_export_preserves_labels_and_rerun_preserves_verified_results(self):
        with tempfile.TemporaryDirectory() as directory, patch('sys.stdout', new_callable=io.StringIO):
            root = Path(directory)
            source = root / 'source with spaces.csv'
            data = dataset()
            SCRIPT.save_csv(source, data, list(data[0]))
            before = source.read_bytes()
            report = SCRIPT.run(source, root)
            audit = root / 'data' / 'processed' / report['run_id'] / 'interval_audit.csv'
            with audit.open(encoding='utf-8-sig', newline='') as handle:
                output = list(csv.DictReader(handle))
            expected = {(r['record_key'], r['interval_index']): r['average_voltage_V'] for r in data}
            self.assertTrue(all(r['average_voltage_V'] == expected[(r['record_key'], r['interval_index'])] for r in output))
            old_bytes = audit.read_bytes()
            repeated = SCRIPT.run(source, root)
            self.assertEqual(report['run_id'], repeated['run_id'])
            self.assertEqual(old_bytes, audit.read_bytes())
            self.assertEqual(before, source.read_bytes())
            audit.write_text('modified by user')
            with self.assertRaisesRegex(ValueError, 'changed or is missing'):
                SCRIPT.run(source, root)
            self.assertEqual(audit.read_text(), 'modified by user')


if __name__ == '__main__':
    unittest.main()
