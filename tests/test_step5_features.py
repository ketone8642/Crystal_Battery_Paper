"""Synthetic tests for shared features, leakage boundaries and portable states."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.features.reaction_features import FEATURE_NAMES, SCHEMA_HASH, ReactionInput, feature_vector, feature_matrix
from src.features.preprocessing import FeatureData, Preprocessor, fit_preprocessor, prepare_fold


def reaction(ion='Li'):
    return {'working_ion': ion, 'formula_charge': 'FePO4', 'formula_discharge': ion + 'FePO4',
            'crystal_system_charge': 'Orthorhombic', 'crystal_system_discharge': 'Orthorhombic',
            'spacegroup_number_charge': '62', 'spacegroup_number_discharge': '62',
            'fracA_charge': '0', 'fracA_discharge': str(1/7)}


def numerical_development():
    # Invented numeric matrices test boundaries; they are not electrode data.
    n = 40
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    i = np.arange(n, dtype=float)
    X[:, 0] = i
    X[:, 1] = i % 3
    X[:, 2] = np.sin(i)
    X[:, 3] = np.cos(i)
    X[:, 4] = (i % 5)**2
    X[:, 5] = 7
    return FeatureData(X, i / 10, np.asarray([f'row-{j}' for j in range(n)]),
                       np.asarray([f'group-{j//2}' for j in range(n)]),
                       np.asarray(['Li'] * n), np.asarray([(j//2) % 5 for j in range(n)]),
                       'development', 'li_only')


class FeatureTests(unittest.TestCase):
    def test_exact_atom_fraction_and_feature_dimensions(self):
        X = feature_vector(reaction())
        self.assertEqual(X.shape, (740,))
        self.assertAlmostEqual(X[FEATURE_NAMES.index('discharge_atom_fraction_Li')], 1/7)
        self.assertAlmostEqual(X[FEATURE_NAMES.index('discharge_atom_fraction_O')], 4/7)
        self.assertAlmostEqual(X[FEATURE_NAMES.index('charge_atom_fraction_O')], 4/6)
        self.assertAlmostEqual(X[:118].sum(), 1)
        self.assertAlmostEqual(X[118:236].sum(), 1)
        self.assertEqual(X[FEATURE_NAMES.index('ion_per_host_discharge')], 1)

    def test_target_source_ids_and_review_flags_cannot_change_features(self):
        original = reaction()
        altered = dict(original, average_voltage_V='-99999', record_key='different', battery_id='test',
                       row_uid='test', group_id='other', partition='holdout', mixed_cv_fold='9',
                       warnings='malicious or target-related text', priority_voltage_review=True,
                       num_steps='900', canonical_host='wrong', framework_formula='wrong')
        np.testing.assert_array_equal(feature_vector(original), feature_vector(altered))

    def test_csv_input_and_prediction_input_generate_identical_features(self):
        csv_row = reaction()
        request = ReactionInput('Li', 'FePO4', 'LiFePO4', 'orthorhombic', 62, 'orthorhombic', 62)
        np.testing.assert_array_equal(feature_vector(csv_row), feature_vector(request))

    def test_formula_multiples_have_identical_features(self):
        scaled = dict(reaction(), formula_charge='Fe2(PO4)2', formula_discharge='Li2Fe2(PO4)2')
        np.testing.assert_allclose(feature_vector(reaction()), feature_vector(scaled), atol=1e-14)

    def test_different_working_ion_changes_composition_and_identity(self):
        li, na = feature_vector(reaction()), feature_vector(reaction('Na'))
        self.assertFalse(np.array_equal(li, na))
        self.assertEqual(na[FEATURE_NAMES.index('working_ion_atomic_number')], 11)
        self.assertAlmostEqual(na[FEATURE_NAMES.index('discharge_atom_fraction_Na')], 1/7)

    def test_inconsistent_input_is_rejected(self):
        changes = [dict(fracA_discharge='0.9'), dict(fracA_charge='-0.0000001'),
                   dict(spacegroup_number_charge='225'), dict(spacegroup_number_charge=True),
                   dict(formula_discharge='LiCoPO4'), dict(formula_discharge='Li0.5FePO4'),
                   dict(working_ion='Rb'), dict(fracA_discharge='nan')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                feature_vector(dict(reaction(), **change))

    def test_empty_matrix_has_fixed_width(self):
        self.assertEqual(feature_matrix([]).shape, (0, 740))


class PreprocessingTests(unittest.TestCase):
    def test_validation_features_are_not_used_for_scaler_or_pca(self):
        base = numerical_development()
        altered = deepcopy(base)
        altered.X[altered.fold == 0, :5] += 1000000
        p1 = fit_preprocessor(base, fold=0, n_components=3)
        p2 = fit_preprocessor(altered, fold=0, n_components=3)
        np.testing.assert_allclose(p1.scale_mean, base.X[base.fold != 0][:, p1.mask].mean(axis=0))
        np.testing.assert_array_equal(p1.components, p2.components)
        np.testing.assert_array_equal(p1.scale_mean, p2.scale_mean)
        self.assertEqual(p1.metadata['training_rows'], 32)
        self.assertEqual(p1.metadata['validation_rows'], 8)

    def test_targets_do_not_affect_preprocessing(self):
        base = numerical_development()
        altered = replace(base, y=np.full(40, -12345.0))
        p1 = fit_preprocessor(base, fold=1, n_components=3)
        p2 = fit_preprocessor(altered, fold=1, n_components=3)
        np.testing.assert_array_equal(p1.components, p2.components)
        np.testing.assert_array_equal(p1.scale, p2.scale)

    def test_holdout_and_transfer_cannot_be_fitted(self):
        for role in ('holdout', 'held_out_Na'):
            data = numerical_development()
            data.role = role
            data.fold = np.full(40, -1)
            if role == 'held_out_Na':
                data.ion = np.asarray(['Na'] * 40)
            with self.assertRaisesRegex(ValueError, 'only be fitted using development'):
                fit_preprocessor(data, n_components=3)

    def test_group_leakage_in_fold_assignment_is_rejected(self):
        data = numerical_development()
        data.fold[1] = 1
        with self.assertRaisesRegex(ValueError, 'spans multiple'):
            fit_preprocessor(data, fold=0, n_components=3)

    def test_full_development_transform_cannot_be_reused_for_cv(self):
        data = numerical_development()
        pp = fit_preprocessor(data, n_components=3)
        with self.assertRaisesRegex(ValueError, 'not fitted on this CV'):
            prepare_fold(data, 0, pp)

    def test_wrong_fold_or_changed_training_matrix_rejects_cached_transform(self):
        data = numerical_development()
        pp = fit_preprocessor(data, fold=0, n_components=3)
        with self.assertRaisesRegex(ValueError, 'not fitted on this CV'):
            prepare_fold(data, 1, pp)
        data.X[data.fold != 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'not fitted on this CV'):
            prepare_fold(data, 0, pp)

    def test_npz_roundtrip_preserves_feature_alignment_and_transforms(self):
        data = numerical_development()
        pp = fit_preprocessor(data, fold=0, n_components=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data.save(root / 'features.npz')
            restored_data = FeatureData.load(root / 'features.npz')
            pp.save(root / 'preprocessor.npz')
            restored_pp = Preprocessor.load(root / 'preprocessor.npz')
            np.testing.assert_array_equal(data.X, restored_data.X)
            np.testing.assert_array_equal(data.row_uid, restored_data.row_uid)
            np.testing.assert_allclose(pp.transform(data.X), restored_pp.transform(data.X), atol=1e-12)
            X_train, y_train, X_val, y_val = prepare_fold(restored_data, 0, restored_pp)
            self.assertEqual((X_train.shape, X_val.shape), ((32, 3), (8, 3)))
            np.testing.assert_array_equal(y_train, data.y[data.fold != 0])
            np.testing.assert_array_equal(y_val, data.y[data.fold == 0])

    def test_schema_mismatch_is_rejected(self):
        pp = fit_preprocessor(numerical_development(), n_components=3)
        pp.metadata['schema_hash'] = 'old-schema'
        with self.assertRaisesRegex(ValueError, 'schema'):
            pp.transform(np.zeros((1, 740)))

    def test_unseen_constant_feature_is_reported_and_not_invented(self):
        data = numerical_development()
        pp = fit_preprocessor(data, n_components=3)
        external = data.X[:2].copy()
        external[0, 30] = 1
        diagnostics = pp.support_diagnostics(external)
        self.assertEqual(diagnostics['rows_with_changed_training_constant_features'], 1)
        self.assertIn(FEATURE_NAMES[30], diagnostics['changed_training_constant_feature_names'])
        np.testing.assert_allclose(pp.transform(external), pp.transform(data.X[:2]))

    def test_insufficient_dimensions_fail_instead_of_padding_components(self):
        with self.assertRaisesRegex(ValueError, 'Not enough training'):
            fit_preprocessor(numerical_development(), n_components=80)


if __name__ == '__main__':
    unittest.main()
