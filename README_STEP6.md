# Step 6: Train and evaluate voltage prediction models

Your Step 5 run completed successfully:

- Mixed ions: 6,140 development intervals and 630 holdout intervals.
- Li only: 3,021 development intervals and 324 holdout intervals.
- Held-out ions: 472 Na intervals and 177 K intervals.
- Each interval has 740 raw features. Each population has ten CV preprocessors
  and a separate full-development preprocessor, with 80 PCA components.

This step trains DNN, SVR and KRR. It produces saved models, evaluation tables,
individual predictions and holdout scatter plots. It does not start or modify
the browser application; model integration follows after reviewing these results.

## 1. Install the addon

Extract `battery_voltage_step6_training.zip`. Merge its `scripts`, `src`, and
`tests` folders into your existing project. Also copy this README. The existing
Step 4 and Step 5 code and data are required. No package upgrade or API key is
needed. Keep `verification` as supporting material rather than copying its
contents over your project reports.

Your project directory is:

```text
D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project
```

Open PowerShell in that directory. The web server may be stopped with Ctrl+C
in its own terminal to free resources; training does not need the server.

## 2. Run the training tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_step6_training.py" -v
```

Expected ending:

```text
Ran 17 tests in ...s

OK
```

These include actual TensorFlow, SVR and KRR fits on small synthetic arrays,
serialization/reloading and an isolated end-to-end training run. The synthetic
data verify software behavior; their metrics are not battery research results.
The temporary test files are removed automatically. Two tests skip if TensorFlow
is missing; both should run in your already-verified Step 2 environment.

The tests also check that:

- CV refuses a preprocessor fitted on the full development set.
- Changed training features invalidate cached preprocessing.
- Holdout data cannot be passed into the CV training entry point.
- Model selection uses the CV score even when unrelated test metrics favor another model.
- Changed checkpoint files fail integrity checks.
- Completed folds resume without fitting again.
- An interruption after the first fold restarts only the unfinished fold.
- Known/new-group transfer metrics are separate, and extreme labels stay in overall metrics.
- Final model selection remains unchanged throughout evaluation.

Local verification used TensorFlow 2.20.0, scikit-learn 1.6.1, NumPy 2.1.3,
SciPy 1.15.3 and joblib 1.5.1, matching your pinned core package versions.
The verification machine runs Linux/Python 3.12; your execution is Windows/Python
3.11. All supplied code was checked for Python 3.11 syntax. Run the tests above
to confirm behavior in your environment. This package does not claim that a
complete Windows training run has already been performed.

## 3. Train all three model families

Use your own completed Step 5 folder name, as printed in your terminal:

```powershell
.\.venv\Scripts\python.exe scripts\step6_train_models.py --step5-run step5_856f632f34_976996a3
```

Leave the terminal running. This performs 280 CV model fits and six final model
fits, sequentially, using both training populations. DNN fits each use 100 epochs.
It can take substantially longer than feature preparation. Runtime depends on
CPU, available RAM and background applications; no Windows runtime is promised.

The DNN prints progress at epoch 1, every 25 epochs and the final epoch. Kernel
methods print when each fold completes. A quiet interval during a kernel fit does
not by itself indicate a hang. KRR uses a dense training kernel and can temporarily
need several hundred MB of memory per fit. Fits run sequentially with limited
numerical threads.

Do not launch a second copy of the same training command while the first is running.

Expected final marker:

```text
STEP 6 TRAINING AND EVALUATION COMPLETE
Report: ...\reports\step6_<run_id>\step6_report.json
Share step6_report.json before connecting models to the web application.
```

The program prints measured holdout/Na/K MAE for each trained family. These
numbers cannot be filled in before your run finishes. There is no promise of
reproducing the paper's approximately 0.4 V holdout MAE.

## Training and model selection protocol

For each population and candidate:

1. Take nine of the saved development folds for training.
2. Verify that the corresponding Step 5 scaler/PCA was fitted on exactly those rows.
3. Transform training and validation features using that training-only state.
4. Fit the model using only the training features and voltage labels.
5. Predict the remaining validation fold, record metrics and save predictions.
6. Repeat for all ten validation folds.

Select the candidate with the lowest mean fold MAE within each model family.
Ties use the candidate ID in lexical order. A separate overall family choice is
also recorded using CV, without consulting holdout results. CV scores used to
choose candidates can be optimistic; final holdout performance is reported separately.

All six choices, across both populations, are written to `selection.json` before
held-out datasets are loaded for final evaluation. Each chosen model is then
fitted on its entire development partition with the matching full-development
preprocessor. The program saves and reloads each bundle and verifies its predictions.

Only after all six final bundles are ready are the holdout, Na and K datasets
evaluated. There is no fitting, epoch selection or hyperparameter tuning in that
evaluation phase. Test labels are never used for target normalization.

## Features and PCA comparison

Each model configuration is tested with both representations:

| Representation | Input to model |
|---|---|
| `pca` | The 80 components saved by Step 5 |
| `scaled` | All columns that vary in the corresponding training partition, standardized with its saved scaler, without PCA |

With no PCA, input width can differ across folds because constant-column removal
is learned separately for each training fold. The network input shape is created
from the actual transformed width. The fixed raw schema remains 740 columns.

For the final mixed-ion fit there are 497 varying raw columns; for Li only there
are 431. The 80-component transformation retained about 43.15% and 51.53% of
standardized feature variance respectively. These percentages are not accuracy.
The CV comparison determines which of the two representations each family uses.
This small initial search does not establish an optimal PCA dimension; studying
additional dimensions would be a separate, predeclared development experiment.

## Exact initial configurations

Every row below is tried with both representations, yielding 14 candidates per
population and 28 across the two populations.

| Family | Configurations per representation |
|---|---|
| SVR | RBF kernel; `(C=10, gamma=0.1)`, `(C=10, gamma=scale)`, `(C=100, gamma=scale)`; epsilon=0.1 V |
| KRR | RBF kernel; `(alpha=0.01, gamma=0.1)`, `(alpha=0.01, gamma=scale)`, `(alpha=1, gamma=scale)` |
| DNN | Input → Dense 60 ReLU → Dropout 0.25 → Dense 30 ReLU → Dropout 0.10 → Dense 1 linear |

For both kernel models, `gamma=scale` is resolved from **only the transformed
training matrix** as `1 / (number_of_columns * variance_of_training_matrix)`.
The numeric value is saved. The KRR implementation explicitly calculates it
because KernelRidge does not accept the SVR string setting directly.

DNN settings are RMSprop with learning rate 0.001, rho 0.9, momentum 0,
epsilon 1e-7 and centered=False; L2 coefficient 0.0001 on the two hidden-layer
kernels; batch size 64; 100 fixed epochs; shuffled training batches. There is
no early stopping. Validation labels do not select the epoch inside a fold.
The final model also trains for 100 epochs. The objective is MSE on standardized
targets plus L2; displayed training objective values are not MAE in volts.

The base seed is 2026. Fold fits use base seed plus fold number; final fits use
base seed plus 1000. TensorFlow deterministic operations and explicit seeds are
enabled. Exact bitwise agreement across machines and library versions is not guaranteed.

### Target transformations

- SVR fits voltages directly in volts, so epsilon=0.1 means 0.1 V.
- KRR subtracts the training-label mean and adds it back after prediction. This
  supplies a training-mean baseline for predictions far from observed inputs.
- DNN subtracts the training-label mean and divides by the training-label standard
  deviation. The saved model reverses this transformation to return volts.

These are documented engineering choices. The original paper did not specify
all settings. Its exact 237 features are unavailable, whereas our adaptation
uses the 740-column representation. RBF KRR, target transformations, ReLU,
learning rate, L2 coefficient, batch size and epoch count must not be presented
as recovered details of the original implementation.

## Outputs

| Location | Contents |
|---|---|
| `reports/step6_<run_id>/step6_report.json` | Complete configuration, CV results, chosen candidates and evaluation metrics |
| `reports/step6_<run_id>/step6_report.md` | Readable model comparison and limitations |
| `reports/step6_<run_id>/model_comparison.csv` | Selected candidate per family, CV variation and final metrics |
| `reports/step6_<run_id>/selection.json` | Choices frozen before final evaluation |
| `reports/step6_<run_id>/cv/` | Completed folds, training histories and out-of-fold predictions |
| `reports/step6_<run_id>/predictions/` | Per-row reference/predicted voltage, absolute error and identifiers |
| `reports/step6_<run_id>/*_holdout.png` | Reference-versus-prediction plots with equal axes |
| `models/<population>/step6_<run_id>/<family>/` | Each final model bundle |

A bundle contains the trained `model.keras` or `model.joblib`, target transformation,
numeric preprocessing, schema hash, software versions, training population,
CV selection information and file checksums. Use only bundles generated by your
own trusted training workflow; joblib files contain serialized Python objects.

The saved kernel models require the recorded scikit-learn version when reloaded.
Keep your working environment and dependency lock file with the project.

## Metrics and interpretation

The program reports:

- MAE and RMSE in volts.
- Direct predictive R2, which can be negative. It is not fitted-line R2.
- Mean signed error, defined as prediction minus reference.
- Per-ion errors within each evaluated partition.
- Negative-reference and priority-review-label subgroup metrics, while retaining
  every row in the overall metric.
- Na/K errors separately for groups observed in development and groups absent
  from development.

The median baseline predicts the median voltage of the relevant training
partition. Its CV predictions use only the nine training folds; final evaluation
uses the full development median. Check that learned models improve on it.

Mean fold MAE gives equal weight to each fold. Pooled out-of-fold MAE weights
each row equally. They can differ slightly when fold sizes differ. Fold standard
deviation is calculated with ddof=1 and is not a prediction confidence interval.
R2 is left null for fewer than two examples or a constant target; empty subgroups
have zero rows and null metrics rather than fabricated scores.

The following transfer group counts are relative to the matching development set:

| Ion | Population | Known group | New group |
|---|---|---:|---:|
| Na | Mixed ions | 343 | 129 |
| K | Mixed ions | 120 | 57 |
| Na | Li only | 325 | 147 |
| K | Li only | 107 | 70 |

These Na/K rows come from the same Materials Project retrieval. They are held-out
ions, not an independent experimental dataset or the paper's original 32-example
sodium set. No fine-tuning uses these rows. A new group means absent under our
grouping rule, not proof of complete chemical novelty.

The 14 previously flagged extreme-voltage rows remain pending source-energy
review. No negative voltage is automatically rejected, no voltage is clipped,
and no label is imputed. The program does not certify chemical stability.

## Resuming after interruption

If you press Ctrl+C, run the **same command** again. Completed, checksum-verified
folds are reused. A fold interrupted before saving is repeated. A completed
final bundle is reused; an interrupted final model fit is restarted. Interrupted
evaluation can be rerun from the same frozen models.

A run's identity includes its inputs, configuration, software versions and code
hashes. Changing these creates a separate run. Do not repeatedly change settings
after inspecting the same holdout results and still describe that holdout as
an untouched final test.

If the process was forcibly closed, a `training.lock` file may remain. Verify
that no training process is running, then remove only the lock path printed in
the error and rerun. Do not delete the CV results or model folders to resume.

## Common errors

| Message or symptom | Action |
|---|---|
| Cannot find `step5_manifest.json` | Use your actual Step 5 run name: `step5_856f632f34_976996a3` |
| `No module named src.training...` | Merge the entire addon `src` folder, including all supplied Python files |
| TensorFlow missing or DLL error | Use the existing `.venv` Python and rerun the Step 2 environment checker |
| File integrity check failed | Share the exact error. Do not edit hashes or suppress the check |
| Not enough RAM during KRR | Stop other memory-heavy applications and retry the same command; do not run fits in parallel |
| Training lock exists | Follow the recovery instructions above after confirming the old process stopped |
| Different run ID from an example | Expected if input manifest, configuration, software or code hashes differ |
| Long silence during a kernel fit | Allow the current fit to finish; progress is printed between folds |
| Poor or negative R2 | This can be a real model result. Inspect MAE, the baseline, labels and per-ion errors |
| Na/K errors exceed familiar-ion errors | Unseen-ion prediction can be substantially harder; inspect known/new-group results |

For a deliberately partial diagnostic run, `--families svr krr` omits DNN, or
`--populations li_only` limits the population. Such a run is clearly recorded
as partial in its configuration and cannot substitute for the complete comparison.
Use the default command for this step.

## Using a saved bundle in the next step

The inference entry point uses the same feature generator used for training:

```python
from src.inference.bundle import VoltageBundle

bundle = VoltageBundle("models/li_only/step6_YOUR_RUN/dnn")
voltages = bundle.predict([{
    "working_ion": "Li",
    "formula_charge": "FePO4",
    "formula_discharge": "LiFePO4",
    "crystal_system_charge": "orthorhombic",
    "crystal_system_discharge": "orthorhombic",
    "spacegroup_number_charge": 62,
    "spacegroup_number_discharge": 62,
}])
print(voltages)
```

This is an input-format example for the specified structure description, not
a promised voltage. Replace the run directory with your completed run. The
current return value is a numerical prediction, without calibrated uncertainty.
The web interface and its model/transfer notices will be added in the next step.

## Reference documentation

- [scikit-learn SVR](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVR.html)
- [scikit-learn KernelRidge](https://scikit-learn.org/1.6/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
- [TensorFlow RMSprop](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)

When the complete command finishes, share `step6_report.json` or the final
console output before proceeding to the web application integration step.
