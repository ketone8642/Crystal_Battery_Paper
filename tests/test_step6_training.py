"""Training guards, real model fits and an isolated synthetic end-to-end run."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.features.preprocessing import FeatureData, Preprocessor, fit_preprocessor
from src.features.reaction_features import FEATURE_NAMES, SCHEMA_HASH
from src.inference.bundle import VoltageBundle
from src.training.experiment import check_preprocessor, cross_validate, evaluate_arrays, run, select_candidate
from src.training.io_utils import read_json, sha, verify_files, write_json
from src.training.voltage_models import candidates, fit_regressor, metrics, Regressor


HAS_TF = importlib.util.find_spec('tensorflow') is not None


def data(role='development', n=24):
    rng = np.random.default_rng(912 if role == 'development' else 419)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=float)
    X[:, :8] = rng.normal(size=(n, 8))
    y = 2.0 + 0.3 * X[:, 0] - 0.1 * X[:, 1]
    gids = np.asarray([f'{role}_g{i//2}' for i in range(n)], dtype='U80')
    folds = np.asarray([(i//2) % 2 if role == 'development' else -1 for i in range(n)])
    ion = role.rsplit('_', 1)[1] if role.startswith('held_out_') else 'Li'
    return FeatureData(X, y, np.asarray([f'{role}_r{i}' for i in range(n)]), gids,
                       np.full(n, ion), folds, role, 'li_only')


def fixture(root):
    run_id = 'step5_synthetic_test'
    base = root / 'data' / 'features' / run_id / 'li_only'
    pp_dir = base / 'preprocessors'
    pp_dir.mkdir(parents=True)
    development = data()
    development.save(base / 'development.npz')
    for role in ('holdout', 'held_out_Na', 'held_out_K'):
        heldout = data(role, 6)
        if role.startswith('held_out_'):
            heldout.group_id[:2] = development.group_id[:2]
        heldout.save(base / (role + '.npz'))
    for fold in (0, 1):
        fit_preprocessor(development, fold=fold, n_components=2).save(pp_dir / f'cv_fold_{fold:02d}.npz')
    fit_preprocessor(development, n_components=2).save(pp_dir / 'all_development.npz')
    report_dir = root / 'reports' / run_id
    report_dir.mkdir(parents=True)
    write_json(report_dir / 'step5_manifest.json', {'status': 'complete',
        'configuration': {'schema_hash': SCHEMA_HASH, 'n_components': 2},
        'output_sha256': {p.relative_to(root).as_posix(): sha(p) for p in base.rglob('*.npz')}})
    return run_id, development, pp_dir


class TrainingTests(unittest.TestCase):
    def test_metrics_are_voltage_errors_and_direct_predictive_r2(self):
        result = metrics(np.array([1., 2., 3.]), np.array([2., 2., 2.]))
        self.assertAlmostEqual(result['mae_V'], 2/3)
        self.assertEqual(result['r2'], 0)
        self.assertLess(metrics(np.array([1., 2., 3.]), np.array([11., 12., 13.]))['r2'], 0)

    def test_empty_and_constant_target_r2_are_not_fabricated(self):
        self.assertIsNone(metrics(np.array([]), np.array([]))['mae_V'])
        self.assertIsNone(metrics(np.ones(3), np.ones(3))['r2'])

    def test_metrics_reject_misaligned_or_nonfinite_predictions(self):
        with self.assertRaises(ValueError):
            metrics(np.ones(3), np.ones(2))
        with self.assertRaises(ValueError):
            metrics(np.ones(3), np.full(3, np.nan))

    def test_candidate_grid_has_all_families_and_both_representations(self):
        grid = candidates()
        self.assertEqual(len(grid), 14)
        self.assertEqual(len(set(c['id'] for c in grid)), 14)
        for family in ('svr', 'krr', 'dnn'):
            self.assertEqual({c['representation'] for c in grid if c['family'] == family}, {'pca', 'scaled'})

    def test_selection_ignores_unrelated_test_metrics(self):
        a = {'candidate': {'id': 'a'}, 'mean_fold_mae_V': 0.2, 'holdout_mae_V': 9.0}
        b = {'candidate': {'id': 'b'}, 'mean_fold_mae_V': 0.4, 'holdout_mae_V': 0.01}
        self.assertEqual(select_candidate([a, b]), a)
        with self.assertRaises(ValueError):
            select_candidate([])

    def test_cv_cannot_use_full_development_preprocessor(self):
        development = data()
        pp = fit_preprocessor(development, n_components=2)
        with self.assertRaises(ValueError):
            check_preprocessor(development, pp, 0)

    def test_cv_checks_training_features_but_does_not_fit_validation_features(self):
        development = data()
        pp = fit_preprocessor(development, fold=0, n_components=2)
        development.X[development.fold == 0, 0] += 99
        check_preprocessor(development, pp, 0)
        development.X[development.fold != 0, 0] += 1
        with self.assertRaises(ValueError):
            check_preprocessor(development, pp, 0)

    def test_cv_rejects_holdout_as_training_data(self):
        heldout = data('holdout')
        with self.assertRaises(ValueError):
            cross_validate(heldout, Path('.'), candidates(), Path('.'), seed=1, epochs=1, batch_size=8)

    def test_real_svr_and_krr_roundtrip_and_training_target_statistics(self):
        development = data()
        X, y = development.X[:, :8], development.y
        for family in ('svr', 'krr'):
            candidate = next(c for c in candidates() if c['family'] == family and c['parameters']['gamma'] == 'scale')
            model = fit_regressor(candidate, X, y)
            self.assertAlmostEqual(model.info['resolved_gamma'], 1/(8*X.var()))
            if family == 'krr':
                self.assertAlmostEqual(model.offset, y.mean())
                self.assertLess(metrics(y, model.predict(X))['mae_V'], 0.05)
            with tempfile.TemporaryDirectory() as temp:
                model.save(temp)
                restored = Regressor.load(temp)
                np.testing.assert_allclose(restored.predict(X), model.predict(X), atol=1e-12)

    @unittest.skipUnless(HAS_TF, 'TensorFlow must be installed to exercise DNN')
    def test_real_tensorflow_fit_and_keras_roundtrip(self):
        development = data()
        candidate = next(c for c in candidates() if c['family'] == 'dnn')
        model = fit_regressor(candidate, development.X[:, :8], development.y, epochs=2, batch_size=8)
        self.assertAlmostEqual(model.offset, development.y.mean())
        self.assertAlmostEqual(model.target_scale, development.y.std())
        self.assertEqual(len(model.info['loss_history']), 2)
        self.assertFalse(model.info['early_stopping'])
        with tempfile.TemporaryDirectory() as temp:
            model.save(temp)
            restored = Regressor.load(temp)
            np.testing.assert_allclose(restored.predict(development.X[:, :8]), model.predict(development.X[:, :8]), rtol=1e-5, atol=1e-5)

    def test_transfer_reports_known_and_new_groups_separately(self):
        development, transfer = data(), data('held_out_Na', 6)
        transfer.group_id[:2] = development.group_id[:2]
        result = evaluate_arrays(development, transfer, transfer.y)
        self.assertEqual(result['known_development_group']['n'], 2)
        self.assertEqual(result['new_development_group']['n'], 4)
        self.assertEqual(result['all']['n'], 6)

    def test_holdout_group_overlap_is_rejected(self):
        development, heldout = data(), data('holdout', 6)
        heldout.group_id[0] = development.group_id[0]
        with self.assertRaises(ValueError):
            evaluate_arrays(development, heldout, heldout.y)

    def test_extreme_labels_are_not_silently_excluded(self):
        development, heldout = data(), data('holdout', 6)
        heldout.y[:2] = [-7., 33.]
        result = evaluate_arrays(development, heldout, np.full(6, 2.))
        self.assertEqual(result['all']['n'], 6)
        self.assertEqual(result['priority_review_reference_voltage']['n'], 2)

    def test_resume_reuses_completed_folds_without_refitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, development, pp_dir = fixture(root)
            grid = [candidates()[0]]
            cache = root / 'cache'
            with redirect_stdout(io.StringIO()):
                a, _ = cross_validate(development, pp_dir, grid, cache, seed=1, epochs=1, batch_size=8)
                with patch('src.training.experiment.fit_regressor', side_effect=AssertionError('refit')):
                    b, _ = cross_validate(development, pp_dir, grid, cache, seed=1, epochs=1, batch_size=8)
            self.assertEqual(a, b)

    def test_changed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, development, pp_dir = fixture(root)
            grid, cache = [candidates()[0]], root / 'cache'
            with redirect_stdout(io.StringIO()):
                cross_validate(development, pp_dir, grid, cache, seed=1, epochs=1, batch_size=8)
            path = cache / grid[0]['id'] / 'fold_00.npz'
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                cross_validate(development, pp_dir, grid, cache, seed=1, epochs=1, batch_size=8)

    def test_interrupted_cv_restarts_only_the_unfinished_fold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, development, pp_dir = fixture(root)
            grid, cache, started = [candidates()[0]], root / 'cache', []

            def interrupted(*args, **kwargs):
                started.append(True)
                if len(started) == 2:
                    raise KeyboardInterrupt()
                return fit_regressor(*args, **kwargs)

            with redirect_stdout(io.StringIO()), patch('src.training.experiment.fit_regressor', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    cross_validate(development, pp_dir, grid, cache, seed=1, epochs=1, batch_size=8)
            self.assertTrue((cache / grid[0]['id'] / 'fold_00.json').exists())
            self.assertFalse((cache / grid[0]['id'] / 'fold_01.json').exists())
            with redirect_stdout(io.StringIO()), patch('src.training.experiment.fit_regressor', wraps=fit_regressor) as fitted:
                cross_validate(development, pp_dir, grid, cache, seed=1, epochs=1, batch_size=8)
                self.assertEqual(fitted.call_count, 1)

    @unittest.skipUnless(HAS_TF, 'TensorFlow must be installed for the complete training workflow')
    def test_end_to_end_all_families_freezes_selection_and_reloads_bundles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            step5_run, _, _ = fixture(root)
            grid = [c for c in candidates() if c['id'].endswith('_0')]
            with redirect_stdout(io.StringIO()):
                report = run(root, step5_run, epochs=2, batch_size=8, populations=('li_only',),
                             grid_override=grid, purpose='synthetic_smoke')
            rdir = root / 'reports' / report['run_id']
            manifest = read_json(rdir / 'step6_manifest.json')
            self.assertEqual(manifest['status'], 'complete')
            verify_files(root, manifest['output_sha256'])
            self.assertEqual(sha(rdir / 'selection.json'), report['selection_sha256_before_evaluation'])
            for relative in report['populations']['li_only']['bundles'].values():
                VoltageBundle(root / relative)
            with redirect_stdout(io.StringIO()), patch('src.training.experiment.fit_regressor', side_effect=AssertionError('refit')):
                repeated = run(root, step5_run, epochs=2, batch_size=8, populations=('li_only',),
                               grid_override=grid, purpose='synthetic_smoke')
            self.assertEqual(report, repeated)


if __name__ == '__main__':
    unittest.main()
