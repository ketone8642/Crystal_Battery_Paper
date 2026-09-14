"""Bounded development-fold verification; not a full model comparison."""
import argparse
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, required=True)
parser.add_argument('--step5-run', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.project_root.resolve()))

import numpy as np
from src.features.preprocessing import FeatureData, Preprocessor, save_numeric_archive
from src.training.experiment import check_preprocessor
from src.training.io_utils import sha, write_json
from src.training.voltage_models import candidates, fit_regressor, metrics, represent, runtime_versions

args.output.mkdir(parents=True, exist_ok=True)
base = args.project_root / 'data' / 'features' / args.step5_run / 'mixed_ions'
development = FeatureData.load(base / 'development.npz')
pp = Preprocessor.load(base / 'preprocessors' / 'cv_fold_00.npz')
train = check_preprocessor(development, pp, 0)
selected = [c for c in candidates() if c['id'].endswith('_0')]
report = {'purpose': 'Software and numerical verification on one development fold only',
          'not_a_full_cv_or_final_test_result': True, 'heldout_datasets_opened': False,
          'source_step5_run': args.step5_run, 'runtime': runtime_versions(),
          'source_development_sha256': sha(base / 'development.npz'),
          'training_rows': int(train.sum()), 'validation_rows': int((~train).sum()),
          'fold': 0, 'models': []}
saved_predictions = {'row_uid': development.row_uid[~train], 'reference_voltage_V': development.y[~train]}
for candidate in selected:
    start = time.perf_counter()
    Xtrain = represent(pp, development.X[train], candidate['representation'])
    Xvalid = represent(pp, development.X[~train], candidate['representation'])
    print('Verifying', candidate['id'], Xtrain.shape, flush=True)
    model = fit_regressor(candidate, Xtrain, development.y[train], epochs=100, seed=2026,
                          progress=lambda x: print(' ', x, flush=True))
    predicted = model.predict(Xvalid)
    result = {'candidate': candidate, 'development_fold_metrics': metrics(development.y[~train], predicted),
              'training_info': model.info, 'elapsed_seconds': time.perf_counter() - start}
    report['models'].append(result)
    saved_predictions[candidate['id']] = predicted
    print('Completed',candidate['id'],flush=True)
save_numeric_archive(args.output / 'development_fold_predictions.npz', **saved_predictions)
write_json(args.output / 'real_fold_verification.json', report)
print('REAL DEVELOPMENT FOLD VERIFICATION COMPLETE', flush=True)
