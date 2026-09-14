"""Grouped CV, frozen selection, final bundles and held-out evaluation."""

import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from src.features.preprocessing import FeatureData, Preprocessor, digest_strings, save_numeric_archive
from src.features.reaction_features import SCHEMA_HASH
from src.inference.bundle import VoltageBundle
from src.training.io_utils import read_json, write_json, sha, verify_files, run_lock
from src.training.voltage_models import candidates, fit_regressor, metrics, represent, runtime_versions, tensorflow


VERSION = 'step6-v1'
POPULATIONS = ('mixed_ions', 'li_only')
FAMILIES = ('svr', 'krr', 'dnn')
LIMITATIONS = [
    'Current Materials Project adaptation, not an exact reproduction of the original paper.',
    'The feature scheme uses composition and symmetry; it has no atomic coordinates, ionic radii or electronegativity.',
    'The search is small and predeclared. Selected CV scores are model-selection statistics, not unbiased test estimates.',
    'All original eligible labels are retained, including the 14 previously flagged priority-review records. Source-energy verification is pending.',
    'Na/K come from the same retrieval and are not the original paper sodium test set or an independent experimental benchmark.',
    'Known-group versus new-group transfer is relative to the development partition used to fit that bundle.',
    'No fine-tuning on Na/K and no calibrated uncertainty estimation are performed.',
    'Li-only direct working-ion identity columns are constant during training and cannot teach a Na/K identity effect.',
    'Metrics are in volts; R2 is computed directly on predictions and can be negative. MAE is not percentage accuracy.',
    'Holdout/transfer results must not be used to choose new hyperparameters while still calling the same sets untouched tests.',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def check_preprocessor(development, preprocessor, fold=None):
    development.validate()
    if development.role != 'development':
        raise ValueError('Model fitting requires development data')
    mask = np.ones(len(development.y), dtype=bool) if fold is None else development.fold != fold
    if fold is not None and fold not in set(development.fold):
        raise ValueError('Unknown validation fold')
    metadata = preprocessor.metadata
    expected_scope = 'all_development' if fold is None else 'cv_training'
    if (metadata['fit_scope'] != expected_scope or metadata['model'] != development.model
            or metadata['validation_fold'] != fold
            or metadata['training_row_uid_sha256'] != digest_strings(development.row_uid[mask])
            or metadata['training_features_sha256'] != hashlib.sha256(np.ascontiguousarray(development.X[mask]).tobytes()).hexdigest()):
        raise ValueError('Preprocessor scope or training partition does not match')
    if set(development.group_id[mask]) & set(development.group_id[~mask]):
        raise ValueError('Training/validation groups overlap')
    return mask


def select_candidate(summaries):
    if not summaries or any(not np.isfinite(item['mean_fold_mae_V']) for item in summaries):
        raise ValueError('Selection needs finite completed CV results')
    # Stable, explicit tie-breaking. No test/transfer argument exists.
    return min(summaries, key=lambda item: (item['mean_fold_mae_V'], item['candidate']['id']))


def cross_validate(development, preprocessor_dir, grid, cache_dir, *, seed, epochs, batch_size):
    if development.role != 'development':
        raise ValueError('Cross-validation requires development data')
    development.validate()
    cache_dir.mkdir(parents=True, exist_ok=True)
    folds = sorted(map(int, set(development.fold)))
    summaries, baseline_oof = [], np.full(len(development.y), np.nan)
    baseline_folds = []
    for fold in folds:
        train, valid = development.fold != fold, development.fold == fold
        baseline_oof[valid] = np.median(development.y[train])
        baseline_folds.append(metrics(development.y[valid], baseline_oof[valid]))
    for candidate in grid:
        candidate_dir = cache_dir / candidate['id']
        candidate_dir.mkdir(exist_ok=True)
        oof = np.full(len(development.y), np.nan)
        fold_reports = []
        for fold in folds:
            valid = development.fold == fold
            state_path = candidate_dir / f'fold_{fold:02d}.json'
            prediction_path = candidate_dir / f'fold_{fold:02d}.npz'
            if state_path.exists():
                state = read_json(state_path)
                if state['candidate'] != candidate or state['fold'] != fold:
                    raise ValueError('CV cache does not match this candidate/fold')
                verify_files(candidate_dir, state['files_sha256'])
                with np.load(prediction_path, allow_pickle=False) as saved:
                    if not np.array_equal(saved['row_uid'], development.row_uid[valid]):
                        raise ValueError('CV cache row order differs')
                    prediction = saved['prediction_V']
                if metrics(development.y[valid], prediction) != state['metrics']:
                    raise ValueError('CV cache metrics differ from its predictions')
                print(f"  {candidate['id']} fold {fold}: resumed; MAE={state['metrics']['mae_V']:.4f} V", flush=True)
            else:
                started = time.perf_counter()
                preprocessor = Preprocessor.load(preprocessor_dir / f'cv_fold_{fold:02d}.npz')
                train = check_preprocessor(development, preprocessor, fold)
                with threadpool_limits(limits=2):
                    Xtrain = represent(preprocessor, development.X[train], candidate['representation'])
                    Xvalid = represent(preprocessor, development.X[valid], candidate['representation'])
                model = fit_regressor(candidate, Xtrain, development.y[train], seed=seed + fold,
                                      epochs=epochs, batch_size=batch_size,
                                      progress=lambda s: print(f"    {candidate['id']} fold {fold}: {s}", flush=True))
                prediction = model.predict(Xvalid)
                score = metrics(development.y[valid], prediction)
                state = {'candidate': candidate, 'fold': fold, 'metrics': score,
                         'training_rows': int(train.sum()), 'validation_rows': int(valid.sum()),
                         'training_groups': len(set(development.group_id[train])),
                         'validation_groups': len(set(development.group_id[valid])),
                         'group_overlap': 0, 'preprocessor_sha256': sha(preprocessor_dir / f'cv_fold_{fold:02d}.npz'),
                         'training_row_uid_sha256': digest_strings(development.row_uid[train]),
                         'elapsed_seconds': time.perf_counter() - started, 'regressor_info': model.info}
                save_numeric_archive(prediction_path, row_uid=development.row_uid[valid], prediction_V=prediction)
                state['files_sha256'] = {prediction_path.name: sha(prediction_path)}
                write_json(state_path, state)
                print(f"  {candidate['id']} fold {fold}: MAE={score['mae_V']:.4f} V", flush=True)
                del model, Xtrain, Xvalid
                gc.collect()
            oof[valid] = prediction
            fold_reports.append(state)
        if not np.isfinite(oof).all():
            raise ValueError('OOF predictions do not cover every development row')
        fold_mae = [state['metrics']['mae_V'] for state in fold_reports]
        summary = {'candidate': candidate, 'folds': len(folds), 'mean_fold_mae_V': float(np.mean(fold_mae)),
                   'std_fold_mae_V': float(np.std(fold_mae, ddof=1)) if len(folds) > 1 else 0.0,
                   'oof_metrics': metrics(development.y, oof), 'fold_results': fold_reports}
        write_json(candidate_dir / 'summary.json', summary)
        save_numeric_archive(candidate_dir / 'oof.npz', row_uid=development.row_uid, prediction_V=oof)
        summaries.append(summary)
    baseline = {'name': 'training_median', 'oof_metrics': metrics(development.y, baseline_oof),
                'mean_fold_mae_V': float(np.mean([s['mae_V'] for s in baseline_folds])),
                'fold_metrics': baseline_folds}
    return summaries, baseline


def make_bundle(development, preprocessor_path, selected, directory, config, versions):
    candidate = selected['candidate']
    if directory.exists():
        bundle = VoltageBundle(directory)
        if bundle.metadata['candidate'] != candidate or bundle.metadata['training_row_uid_sha256'] != digest_strings(development.row_uid):
            raise ValueError('Existing bundle does not match the selected model')
        return bundle
    preprocessor = Preprocessor.load(preprocessor_path)
    check_preprocessor(development, preprocessor)
    with threadpool_limits(limits=2):
        X = represent(preprocessor, development.X, candidate['representation'])
    print(f"Fitting final {development.model}/{candidate['family']}: {candidate['id']}", flush=True)
    model = fit_regressor(candidate, X, development.y, seed=config['seed'] + 1000,
                          epochs=config['epochs'], batch_size=config['batch_size'],
                          progress=lambda s: print(f'  final fit: {s}', flush=True))
    temporary = directory.with_name(directory.name + '.pending_' + uuid4().hex[:8])
    temporary.mkdir(parents=True)
    try:
        model.save(temporary)
        shutil.copy2(preprocessor_path, temporary / 'preprocessor.npz')
        metadata = {'status': 'complete', 'schema_hash': SCHEMA_HASH, 'population': development.model,
                    'candidate': candidate, 'training_rows': len(development.y),
                    'training_ions': sorted(set(map(str, development.ion))),
                    'training_row_uid_sha256': digest_strings(development.row_uid),
                    'fit_scope': 'all_development', 'runtime': versions,
                    'cv_mean_fold_mae_V': selected['mean_fold_mae_V'], 'limitations': LIMITATIONS,
                    'files_sha256': {p.name: sha(p) for p in temporary.iterdir() if p.is_file()}}
        write_json(temporary / 'bundle.json', metadata)
        restored = VoltageBundle(temporary)
        np.testing.assert_allclose(restored.predict_matrix(development.X[:32]), model.predict(X[:32]), rtol=1e-5, atol=1e-5)
        temporary.replace(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    del model, X
    gc.collect()
    return VoltageBundle(directory)


def evaluate_arrays(development, data, prediction):
    """No fitting or selection occurs here; all subgroups remain in main metrics."""
    if data.role == 'development':
        raise ValueError('Final evaluation requires held-out data')
    data.validate()
    shared = np.isin(data.group_id, development.group_id)
    if data.role == 'holdout' and np.any(shared):
        raise ValueError('Development/holdout groups overlap')
    output = {'all': metrics(data.y, prediction),
              'per_ion': {str(ion): metrics(data.y[data.ion == ion], prediction[data.ion == ion]) for ion in sorted(set(data.ion))},
              'negative_reference_voltage': metrics(data.y[data.y < 0], prediction[data.y < 0]),
              'priority_review_reference_voltage': metrics(data.y[(data.y < -3) | (data.y > 10)], prediction[(data.y < -3) | (data.y > 10)])}
    if data.role.startswith('held_out_'):
        output['known_development_group'] = metrics(data.y[shared], prediction[shared])
        output['new_development_group'] = metrics(data.y[~shared], prediction[~shared])
    return output


def write_predictions(path, data, prediction, development):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_uid', 'group_id', 'working_ion', 'role', 'reference_voltage_V',
                         'predicted_voltage_V', 'absolute_error_V', 'group_seen_in_development'])
        seen = set(development.group_id)
        for uid, gid, ion, actual, predicted in zip(data.row_uid, data.group_id, data.ion, data.y, prediction):
            writer.writerow([uid, gid, ion, data.role, actual, predicted, abs(actual-predicted), gid in seen])


def plot_holdout(path, data, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.5), squeeze=False)
    for ax, (family, predicted) in zip(axes[0], predictions.items()):
        lower = float(min(data.y.min(), predicted.min()))
        upper = float(max(data.y.max(), predicted.max()))
        margin = max((upper - lower) * 0.05, 0.1)
        ax.scatter(data.y, predicted, s=10, alpha=0.45, color='#176b87')
        ax.plot([lower-margin, upper+margin], [lower-margin, upper+margin], color='#555555', linestyle='--', linewidth=1)
        ax.set(xlim=(lower-margin, upper+margin), ylim=(lower-margin, upper+margin),
               xlabel='Reference voltage (V)', ylabel='Predicted voltage (V)',
               title=f'{family.upper()} | MAE {metrics(data.y, predicted)["mae_V"]:.3f} V')
        ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report_markdown(report):
    lines = ['# Step 6 training results', '', f"Run: `{report['run_id']}`", '',
             'All candidates were ranked by mean grouped cross-validation MAE. Selection was frozen',
             'before final evaluation. No holdout or Na/K labels were used for fitting or selection.', '',
             '| Population | Family | Chosen representation | CV MAE (V) | Holdout MAE (V) | Na MAE (V) | K MAE (V) |',
             '|---|---|---|---:|---:|---:|---:|']
    for population, item in report['populations'].items():
        for family, selected in item['selected'].items():
            e = item['evaluation'][family]
            lines.append(f"| {population} | {family} | {selected['candidate']['representation']} | {selected['mean_fold_mae_V']:.4f} | {e['holdout']['all']['mae_V']:.4f} | {e['held_out_Na']['all']['mae_V']:.4f} | {e['held_out_K']['all']['mae_V']:.4f} |")
        lines.extend(['', f"{population}: CV-selected family = **{item['cv_selected_family']}**. "
                      f"Median baseline holdout MAE = {item['evaluation']['median_baseline']['holdout']['all']['mae_V']:.4f} V.", ''])
    lines.extend(['', 'See step6_report.json for RMSE, direct predictive R2, bias, per-ion metrics,',
                  'known/new-group Na/K transfer and priority-review/negative-voltage subgroups.', '',
                  'CV fold standard deviations describe fold variation, not prediction intervals.', '',
                  '## Limitations', ''] + ['- ' + line for line in LIMITATIONS])
    return '\n'.join(lines) + '\n'


def print_summary(report, path):
    for population, item in report['populations'].items():
        print(f'\n{population}:')
        for family in item['selected']:
            evaluation = item['evaluation'][family]
            print(f"  {family.upper()}: holdout MAE={evaluation['holdout']['all']['mae_V']:.4f} V; "
                  f"Na MAE={evaluation['held_out_Na']['all']['mae_V']:.4f} V; K MAE={evaluation['held_out_K']['all']['mae_V']:.4f} V")
        print(f"  Family selected by development CV: {item['cv_selected_family']}")
    print('STEP 6 TRAINING AND EVALUATION COMPLETE')
    print(f'Report: {path}')
    print('Share step6_report.json before connecting models to the web application.')


def run(root, step5_run, *, epochs=100, seed=2026, batch_size=64,
        families=FAMILIES, populations=POPULATIONS, grid_override=None, purpose='research'):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', step5_run):
        raise ValueError('Use a Step 5 folder name, not a path')
    if epochs < 1 or batch_size < 1 or seed < 0 or seed > 2**31-2000:
        raise ValueError('Invalid epochs, batch size or seed')
    families, populations = tuple(families), tuple(populations)
    if not families or not set(families) <= set(FAMILIES) or len(set(families)) != len(families):
        raise ValueError('Invalid model families')
    if not populations or not set(populations) <= set(POPULATIONS) or len(set(populations)) != len(populations):
        raise ValueError('Invalid training populations')
    if grid_override is not None and purpose != 'synthetic_smoke':
        raise ValueError('Custom tiny test grids are restricted to the synthetic smoke test')
    source_path = root / 'reports' / step5_run / 'step5_manifest.json'
    source = read_json(source_path)
    if source.get('status') != 'complete' or source['configuration']['schema_hash'] != SCHEMA_HASH:
        raise ValueError('Step 5 must be complete with the current feature schema')
    verify_files(root, source['output_sha256'])
    feature_root = root / 'data' / 'features' / step5_run
    # Guard that required files belong to the verified source, not just its directory.
    def required(path):
        if path.relative_to(root).as_posix() not in source['output_sha256']:
            raise ValueError(f'Required file is not covered by the Step 5 manifest: {path}')
        return path
    development_sets = {}
    for population in populations:
        directory = feature_root / population
        development = FeatureData.load(required(directory / 'development.npz'))
        if development.role != 'development' or development.model != population:
            raise ValueError('Development file contains the wrong role/population')
        development_sets[population] = development
        folds = sorted(set(map(int, development.fold)))
        if folds != list(range(len(folds))) or len(folds) < 2 or (purpose == 'research' and len(folds) != 10):
            raise ValueError('Expected the saved ten grouped validation folds')
        for fold in folds:
            pp = Preprocessor.load(required(directory / 'preprocessors' / f'cv_fold_{fold:02d}.npz'))
            check_preprocessor(development, pp, fold)
        check_preprocessor(development, Preprocessor.load(required(directory / 'preprocessors' / 'all_development.npz')))
        for role in ('holdout', 'held_out_Na', 'held_out_K'):
            required(directory / (role + '.npz'))
    grid = [candidate for candidate in (grid_override or candidates()) if candidate['family'] in families]
    if set(c['family'] for c in grid) != set(families) or len(set(c['id'] for c in grid)) != len(grid):
        raise ValueError('Candidate grid is incomplete or repeats IDs')
    versions = runtime_versions('dnn' in families)
    if 'dnn' in families:
        tensorflow()  # Fail before expensive CV if the installed runtime cannot import.
    code_files = [Path(__file__), Path(__file__).with_name('voltage_models.py'), Path(__file__).with_name('io_utils.py'),
                  Path(__file__).parents[1] / 'inference' / 'bundle.py',
                  Path(__file__).parents[1] / 'features' / 'preprocessing.py',
                  Path(__file__).parents[1] / 'features' / 'reaction_features.py',
                  Path(__file__).parents[1] / 'data' / 'chemistry.py']
    config = {'version': VERSION, 'source_manifest_sha256': sha(source_path), 'epochs': epochs,
              'seed': seed, 'batch_size': batch_size, 'families': list(families), 'populations': list(populations),
              'candidates': grid, 'runtime': versions, 'purpose': purpose,
              'code_sha256': {p.name: sha(p) for p in code_files}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run_id = 'step6_' + signature[:12]
    report_dir = root / 'reports' / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / 'step6_manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['configuration'] != config:
            raise ValueError('Run configuration differs from saved state')
        if old['status'] == 'complete':
            verify_files(root, old['output_sha256'])
            report = read_json(report_dir / 'step6_report.json')
            print('Existing completed run verified. No models were retrained.')
            print_summary(report, report_dir / 'step6_report.json')
            return report
    with run_lock(report_dir):
        manifest = {'status': 'running', 'configuration': config, 'started_at_utc': now()}
        write_json(manifest_path, manifest)
        try:
            report = {'run_id': run_id, 'step5_run': step5_run, 'configuration': config,
                      'populations': {}, 'limitations': LIMITATIONS, 'started_at_utc': now()}
            print(f'Training run: {run_id}', flush=True)
            print(f'{len(grid)} candidates per population; completed folds will be reused on resume.', flush=True)
            for population in populations:
                print(f'\nCross-validating {population}...', flush=True)
                summaries, baseline = cross_validate(development_sets[population],
                    feature_root / population / 'preprocessors', grid, report_dir / 'cv' / population,
                    seed=seed, epochs=epochs, batch_size=batch_size)
                selected = {family: select_candidate([s for s in summaries if s['candidate']['family'] == family])
                            for family in families}
                report['populations'][population] = {'candidate_results': summaries, 'cv_baseline': baseline,
                    'selected': selected, 'cv_selected_family': select_candidate(list(selected.values()))['candidate']['family']}
            # Freeze every population/family choice before opening any held-out dataset.
            selection_path = report_dir / 'selection.json'
            selection = {'selection_metric': 'mean_grouped_validation_fold_MAE_V', 'configuration_sha256': signature,
                         'selection_uses_holdout_or_transfer': False,
                         'selected': {p: {f: s['candidate'] for f, s in item['selected'].items()} for p, item in report['populations'].items()},
                         'cv_selected_family': {p: i['cv_selected_family'] for p, i in report['populations'].items()}}
            if selection_path.exists() and read_json(selection_path) != selection:
                raise ValueError('Frozen selection differs from current CV results')
            write_json(selection_path, selection)
            print('\nAll model selections frozen from development CV.', flush=True)
            for population in populations:
                development = development_sets[population]
                item = report['populations'][population]
                item['bundles'] = {}
                for family, selected in item['selected'].items():
                    directory = root / 'models' / population / run_id / family
                    make_bundle(development, feature_root / population / 'preprocessors' / 'all_development.npz',
                                selected, directory, config, versions)
                    item['bundles'][family] = directory.relative_to(root).as_posix()
            report['evaluation_started_at_utc'] = now()
            report['selection_sha256_before_evaluation'] = sha(selection_path)
            print('\nEvaluating frozen bundles on holdout and held-out ions...', flush=True)
            for population in populations:
                item, development = report['populations'][population], development_sets[population]
                item['evaluation'] = {family: {} for family in list(families) + ['median_baseline']}
                prediction_dir = report_dir / 'predictions' / population
                prediction_dir.mkdir(parents=True, exist_ok=True)
                for role in ('holdout', 'held_out_Na', 'held_out_K'):
                    data = FeatureData.load(feature_root / population / (role + '.npz'))
                    if data.model != population or data.role != role:
                        raise ValueError('Evaluation file contains the wrong role/population')
                    plot_predictions = {}
                    for family in families:
                        bundle = VoltageBundle(root / item['bundles'][family])
                        prediction = bundle.predict_matrix(data.X)
                        evaluation = evaluate_arrays(development, data, prediction)
                        evaluation['support_diagnostics'] = bundle.preprocessor.support_diagnostics(data.X)
                        item['evaluation'][family][role] = evaluation
                        write_predictions(prediction_dir / f'{family}_{role}.csv', data, prediction, development)
                        plot_predictions[family] = prediction
                        del bundle
                        gc.collect()
                    baseline_prediction = np.full(len(data.y), np.median(development.y))
                    item['evaluation']['median_baseline'][role] = evaluate_arrays(development, data, baseline_prediction)
                    write_predictions(prediction_dir / f'median_baseline_{role}.csv', data, baseline_prediction, development)
                    if role == 'holdout' and len(data.y):
                        plot_holdout(report_dir / f'{population}_holdout.png', data, plot_predictions)
            if sha(selection_path) != report['selection_sha256_before_evaluation']:
                raise ValueError('Frozen selection changed during evaluation')
            report['finished_at_utc'] = now()
            write_json(report_dir / 'step6_report.json', report)
            (report_dir / 'step6_report.md').write_text(report_markdown(report), encoding='utf-8')
            with (report_dir / 'model_comparison.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['population', 'family', 'candidate', 'cv_mean_mae_V', 'cv_std_mae_V',
                                 'holdout_mae_V', 'holdout_rmse_V', 'holdout_r2', 'Na_mae_V', 'K_mae_V'])
                for population, item in report['populations'].items():
                    for family, selected in item['selected'].items():
                        evaluation = item['evaluation'][family]
                        writer.writerow([population, family, selected['candidate']['id'], selected['mean_fold_mae_V'],
                            selected['std_fold_mae_V'], evaluation['holdout']['all']['mae_V'], evaluation['holdout']['all']['rmse_V'],
                            evaluation['holdout']['all']['r2'], evaluation['held_out_Na']['all']['mae_V'], evaluation['held_out_K']['all']['mae_V']])
            outputs = [p for p in report_dir.rglob('*') if p.is_file() and p.name not in ('step6_manifest.json', 'training.lock')]
            for item in report['populations'].values():
                for relative in item['bundles'].values():
                    outputs.extend(p for p in (root / relative).rglob('*') if p.is_file())
            manifest.update({'status': 'complete', 'finished_at_utc': now(),
                             'output_sha256': {p.relative_to(root).as_posix(): sha(p) for p in outputs}})
            write_json(manifest_path, manifest)
        except BaseException as exc:
            manifest.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                             'failure_type': type(exc).__name__})
            write_json(manifest_path, manifest)
            raise
    print_summary(report, report_dir / 'step6_report.json')
    return report
