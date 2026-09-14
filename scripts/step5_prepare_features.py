"""Create fixed reaction features and 10-fold plus development-only preprocessors.

Use the existing Python 3.11 environment. No API key or new package is needed.
No voltage prediction model is trained and no prediction error is evaluated.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.reaction_features import SCHEMA, SCHEMA_HASH, FEATURE_NAMES, feature_vector
from src.features.preprocessing import FeatureData, Preprocessor, fit_preprocessor, prepare_fold


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def summary(report, report_dir):
    print(f"Raw features per interval: {report['raw_feature_count']}")
    print(f"PCA components: {report['n_components']}")
    for model, info in report['models'].items():
        print(f"{model}: development={info['datasets']['development']['rows']}, holdout={info['datasets']['holdout']['rows']}")
        print(f"  Full-development PCA variance retained: {info['development_preprocessor']['explained_variance_ratio_sum']:.2%}")
        print(f"  CV preprocessors: {len(info['cv_preprocessors'])}; separate development preprocessor: 1")
    print('STEP 5 FEATURE PREPARATION COMPLETE')
    print(f'Report: {report_dir / "step5_report.json"}')
    print('No voltage prediction model has been trained. Share this report before Step 6.')


def report_text(report):
    lines = ['# Step 5 feature preparation results', '',
             'This is a measured preprocessing report, not a prediction-performance report.', '',
             f"Feature schema: `{SCHEMA['version']}`. Raw features: {len(FEATURE_NAMES)}.",
             'The exact paper feature specification is unavailable; this representation is a documented adaptation.', '',
             '| Feature block | Columns |', '|---|---:|']
    for name, count in report['feature_blocks'].items():
        lines.append(f'| {name} | {count} |')
    lines.extend(['', '| Model | Development | Holdout | Active features in development | PCA variance retained |', '|---|---:|---:|---:|---:|'])
    for name, info in report['models'].items():
        pp = info['development_preprocessor']
        lines.append(f"| {name} | {info['datasets']['development']['rows']} | {info['datasets']['holdout']['rows']} | {pp['active_columns']} | {pp['explained_variance_ratio_sum']:.2%} |")
    lines.extend(['', 'Variance retention is measured after constant-column removal and standardization.',
                  'It is not accuracy, explained target variation, or evidence that PCA improves prediction.',
                  'Retain the raw features and compare no-PCA, 80-component and other training-selected',
                  'representations using the saved validation folds before evaluating holdout.', '',
                  'Each CV scaler/PCA sees only the nine training folds. The separate development',
                  'scaler/PCA sees only the full development partition. Holdout and Na/K feature',
                  'matrices are transformed only; their values and labels are not used to fit preprocessing.', '',
                  'Saved scalers and PCA components use numeric NPZ state, without pickled sklearn',
                  'objects. The portable transform was compared with sklearn during every fit.', '',
                  'The categorical vocabulary is fixed before reading the data: 118 elements,',
                  'eight working ions, seven crystal systems, and 230 space groups for each endpoint.',
                  'Atomic numbers follow the',
                  '[IUPAC periodic-table order](https://iupac.org/what-we-do/periodic-table-of-elements/).', '',
                  'Standardization learns training means and scales as described by',
                  '[StandardScaler](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html).',
                  'Full-SVD PCA is applied without whitening; PCA itself centers inputs but does not scale features,',
                  'as described in the [PCA documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html).', '',
                  '## Limits retained from the data and representation', ''])
    lines.extend('- ' + value for value in report['limitations'])
    lines.extend(['', 'No neural network, SVR or KRR was fitted. No MAE, RMSE or R-squared is claimed.'])
    return '\n'.join(lines) + '\n'


def run(project_root, step4_run, n_components=80):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', step4_run):
        raise ValueError('Use the Step 4 run folder name, not a path')
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError('PCA components must be a positive integer')
    step4_reports = project_root / 'reports' / step4_run
    source_manifest_path = step4_reports / 'step4_manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    if source_manifest.get('status') != 'complete':
        raise ValueError('Step 4 is incomplete')
    # Verify the source run before reading its matrices and fold assignments.
    for relative, expected in source_manifest['output_sha256'].items():
        path = (project_root / relative).resolve()
        if not path.is_relative_to(project_root.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Step 4 integrity check failed: {relative}')
    configuration = {'algorithm_version': 'step5-v1.1', 'schema_hash': SCHEMA_HASH,
                     'source_manifest_sha256': sha(source_manifest_path), 'n_components': n_components}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    run_id = 'step5_' + source_manifest['input_sha256'][:10] + '_' + config_hash[:8]
    feature_dir = project_root / 'data' / 'features' / run_id
    report_dir = project_root / 'reports' / run_id
    manifest_path = report_dir / 'step5_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'complete':
            raise ValueError(f'Incomplete Step 5 run exists at {report_dir}; preserve it and share the error')
        for rel, expected in manifest['output_sha256'].items():
            path = project_root / rel
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f'Previous Step 5 output was changed or is missing: {rel}')
        report = json.loads((report_dir / 'step5_report.json').read_text(encoding='utf-8'))
        print('Verified existing features and preprocessors for this source run.')
        summary(report, report_dir)
        return report
    if feature_dir.exists() or report_dir.exists():
        raise ValueError('An output folder exists without a complete manifest; it was not overwritten')
    feature_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest = {'status': 'running', 'configuration': configuration, 'output_sha256': {}}
    write_json(manifest_path, manifest)
    outputs = []
    report = {
        'run_id': run_id, 'source_step4_run': step4_run,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_feature_count': len(FEATURE_NAMES), 'feature_schema_hash': SCHEMA_HASH,
        'n_components': n_components,
        'feature_blocks': dict(Counter(column['block'] for column in SCHEMA['columns'])),
        'voltage_prediction_model_trained': False, 'prediction_metrics_computed': False,
        'models': {}, 'limitations': [
            'The fixed 740-column representation is an adaptation, not the unavailable original 237-feature implementation.',
            'Composition and symmetry do not specify atomic coordinates or independently validate voltage labels.',
            'The 14 priority voltage-review records retain their original labels; source-energy verification remains pending.',
            'Atomic-number summaries are simple descriptors; electronegativity, ionic radii and learned crystal representations are not included.',
            'Standardizing rare categorical columns can spread variance over many components. PCA dimensionality must be assessed through validation, not assumed adequate.',
            'The working ion never varies within Li-only training. Its direct ion-identity columns are training-constant and removed; zero-shot Na/K performance is unverified.',
            'Changed training-constant features are reported on held-out data and are not learned by the fitted projection. Such diagnostics are not calibrated prediction uncertainty.',
            'Na/K are held-out ions from the same database; known-group and new-group transfer must be evaluated separately.',
            'Numerical fits can differ slightly across Python, NumPy, sklearn and BLAS versions. Each model must use its own saved matching preprocessor.',
        ],
    }
    try:
        schema_path = report_dir / 'feature_schema.json'
        write_json(schema_path, {**SCHEMA, 'schema_hash': SCHEMA_HASH})
        outputs.append(schema_path)
        for model in ('mixed_ions', 'li_only'):
            print(f'\nPreparing {model} features...', flush=True)
            model_dir = feature_dir / model
            model_dir.mkdir()
            pp_dir = model_dir / 'preprocessors'
            pp_dir.mkdir()
            sources = {
                'development': project_root / 'data' / 'splits' / step4_run / model / 'development.csv',
                'holdout': project_root / 'data' / 'splits' / step4_run / model / 'holdout.csv',
                'held_out_Na': project_root / 'data' / 'splits' / step4_run / 'held_out_ions' / 'Na.csv',
                'held_out_K': project_root / 'data' / 'splits' / step4_run / 'held_out_ions' / 'K.csv',
            }
            datasets, info = {}, {'datasets': {}, 'cv_preprocessors': []}
            for role, path in sources.items():
                data = FeatureData.from_rows(read_csv(path), role, model)
                data_path = model_dir / (role + '.npz')
                data.save(data_path)
                outputs.append(data_path)
                datasets[role] = data
                info['datasets'][role] = {'rows': len(data.X), 'features': data.X.shape[1],
                                          'source_csv_sha256': sha(path)}
                print(f'  {role}: {data.X.shape[0]} x {data.X.shape[1]}', flush=True)
            development = datasets['development']
            heldout = datasets['holdout']
            if set(development.group_id) & set(heldout.group_id):
                raise ValueError('Development/holdout groups overlap')
            folds = sorted(set(development.fold))
            if len(folds) != source_manifest['configuration']['cv_folds'] or folds != list(range(len(folds))):
                raise ValueError('Development folds do not match the Step 4 configuration')
            for fold in folds:
                pp = fit_preprocessor(development, fold=int(fold), n_components=n_components)
                pp_path = pp_dir / f'cv_fold_{fold:02d}.npz'
                pp.save(pp_path)
                outputs.append(pp_path)
                # Exercise validation transform and verify saved-scope matching.
                _, _, X_val, _ = prepare_fold(development, int(fold), pp)
                if X_val.shape[1] != n_components:
                    raise ValueError('Unexpected PCA output width')
                info['cv_preprocessors'].append(pp.metadata)
                print(f'  CV fold {fold}: fitted on {pp.metadata["training_rows"]} rows; variance retained {pp.metadata["explained_variance_ratio_sum"]:.2%}', flush=True)
            final_pp = fit_preprocessor(development, n_components=n_components)
            pp_path = pp_dir / 'all_development.npz'
            final_pp.save(pp_path)
            outputs.append(pp_path)
            restored = Preprocessor.load(pp_path)
            info['development_preprocessor'] = final_pp.metadata
            info['support_diagnostics'] = {}
            for role, data in datasets.items():
                transformed = restored.transform(data.X)
                if transformed.shape != (len(data.X), n_components):
                    raise ValueError('Unexpected transformed shape')
                info['datasets'][role]['pca_output_shape'] = list(transformed.shape)
                info['support_diagnostics'][role] = restored.support_diagnostics(data.X)
            report['models'][model] = info
            print(f'  Development-only transform ready; PCA variance retained {final_pp.metadata["explained_variance_ratio_sum"]:.2%}', flush=True)
        report['preprocessing_fit_count'] = sum(len(i['cv_preprocessors']) + 1 for i in report['models'].values())
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report_path = report_dir / 'step5_report.json'
        write_json(report_path, report)
        outputs.append(report_path)
        text_path = report_dir / 'step5_report.md'
        text_path.write_text(report_text(report), encoding='utf-8')
        outputs.append(text_path)
        manifest['status'] = 'complete'
        manifest['output_sha256'] = {p.relative_to(project_root).as_posix(): sha(p) for p in outputs}
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['failure_type'] = type(exc).__name__
        write_json(manifest_path, manifest)
        raise
    summary(report, report_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step4-run', required=True, help='Completed Step 4 run folder name')
    parser.add_argument('--pca-components', type=int, default=80)
    args = parser.parse_args()
    try:
        run(ROOT, args.step4_run, args.pca_components)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'[ERROR] {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
