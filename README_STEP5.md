# Step 5: reaction features and preprocessing

Your Step 4 split is the input to this step. This addon creates numeric features
for the mixed-ion and Li-only experiments and prepares scaling/PCA transforms
for their saved validation folds. It does not train a voltage prediction model.

## Reproduction status

The original paper's full 237-feature specification has not been recovered.
This addon deliberately uses a documented 740-column composition/symmetry
representation. It must be described as an adaptation, not an exact reproduction.
The 80 PCA components are an initial paper-style setting, not a proven optimum
for this different representation.

## Install and run on Windows

1. Extract `battery_voltage_step5_features.zip`.
2. Copy `scripts`, `src`, `tests`, and `README_STEP5.md` into your existing
   `battery_voltage_project` folder. Merge folders. The existing Step 4 modules
   are required and remain in use.
3. Keep `reference_results` as a reference copy of the actual results generated
   here. Run the command below to prepare the corresponding working files on
   your own computer.

No API key or new package installation is required. The code uses NumPy,
scikit-learn and its existing threadpoolctl dependency. TensorFlow is not loaded.

In the existing project PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe scripts\step5_prepare_features.py --step4-run step4_856f632f34_01ddf632
```

Leave it running while it prints the progress of the 22 preprocessing fits.
It uses CPU execution with a small numerical thread limit.

Expected ending:

```text
Raw features per interval: 740
PCA components: 80
mixed_ions: development=6140, holdout=630
  Full-development PCA variance retained: 43.15%
  CV preprocessors: 10; separate development preprocessor: 1
li_only: development=3021, holdout=324
  Full-development PCA variance retained: 51.53%
  CV preprocessors: 10; separate development preprocessor: 1
STEP 5 FEATURE PREPARATION COMPLETE
```

Small numerical differences across library/platform versions are possible. The
reference run used Python 3.12, NumPy 2.3.5 and scikit-learn 1.8.0; the code uses
APIs available in your pinned scikit-learn 1.6.1 and was syntax-checked for Python
3.11. Confirm the live Windows result through the command above.

The Step 5 run folder name can differ from the reference run because it includes
the hash of your locally created Step 4 manifest. Use the actual path printed
in your terminal; the row counts and feature schema should match.

## What the 740 numbers describe

| Feature block | Columns | Meaning |
|---|---:|---|
| Endpoint elemental fractions | 236 | 118 fixed element fractions for each endpoint |
| Endpoint composition summaries | 14 | Seven summaries for each endpoint |
| Working-ion descriptors | 10 | Eight identity indicators, atomic number and period |
| Concentration interval | 6 | Two atom fractions, two loadings per reduced host, and their differences |
| Crystal-system indicators | 14 | Seven categories for each endpoint |
| Space-group indicators | 460 | 230 categories for each endpoint |
| Total | 740 | Fixed feature order for every row |

The seven composition summaries are atom-fraction-weighted mean and standard
deviation of atomic number, minimum/maximum atomic number, distinct-element
count, atom-fraction entropy and atom-fraction L2 norm. Atomic numbers follow
the [IUPAC periodic-table order](https://iupac.org/what-we-do/periodic-table-of-elements/).
No electronegativity, atomic-mass or ionic-radius table is invented or imputed.

One-hot categorical indicators contain either 0 or 1. For example,
`charge_spacegroup_062 = 1` means the charge endpoint has space-group number 62.
Space-group 225 is not encoded as a numeric magnitude that is 225/62 times 62.

The complete ordered dictionary, including each column index and definition,
is saved in `feature_schema.json`.

For `FePO4` to `LiFePO4`, with working ion Li:

| Example feature | Value |
|---|---:|
| Charge oxygen atom fraction | 4/6 = 0.666667 |
| Discharge lithium atom fraction | 1/7 = 0.142857 |
| Discharge oxygen atom fraction | 4/7 = 0.571429 |
| Working-ion atomic number | 3 |
| Charge ion loading per reduced FePO4 host | 0 |
| Discharge ion loading per reduced FePO4 host | 1 |

An atom fraction and a coefficient per host formula are different quantities.
The formulas determine both. Optional supplied `fracA` values are checked against
the calculated atom fractions using a 0.000001 tolerance.

## Exact predictor input contract

The shared feature function reads only:

- `working_ion`
- `formula_charge`, `formula_discharge`
- `crystal_system_charge`, `spacegroup_number_charge`
- `crystal_system_discharge`, `spacegroup_number_discharge`
- Optional `fracA_charge`, `fracA_discharge` consistency checks

Both endpoint symmetry descriptions are required for this baseline. The formula
parser retains the Step 4 restriction to integer stoichiometries and nested
parentheses/brackets. It rejects unsupported notation, inconsistent hosts,
non-increasing loading and inconsistent symmetry rather than fabricating inputs.

Targets, identifiers, grouping/fold assignments, source warnings, review flags,
number of voltage steps and CSV provenance columns cannot become predictors.
The known voltage is stored separately as `y`, in volts.

For a future API request, the same function is used directly:

```python
from src.features.reaction_features import feature_vector

request = {
    'working_ion': 'Li',
    'formula_charge': 'FePO4',
    'formula_discharge': 'LiFePO4',
    'crystal_system_charge': 'Orthorhombic',
    'spacegroup_number_charge': 62,
    'crystal_system_discharge': 'Orthorhombic',
    'spacegroup_number_discharge': 62,
}
X_one = feature_vector(request).reshape(1, -1)  # shape: (1, 740)
```

No voltage label or Materials Project ID is needed to construct this vector.
This code does not predict a voltage; a trained model will be added later.

## Training-only preprocessing

Each preprocessing fit performs these steps:

1. Select only the permitted development training rows.
2. Remove columns that are constant within those rows, using a range tolerance
   of 1e-12. The mask is learned from those training rows only.
3. Fit StandardScaler to the remaining columns.
4. Fit full-SVD PCA with 80 components and no whitening.
5. Save the feature mask, means, scales, PCA mean and projection components.
6. Apply the saved transform to validation or held-out inputs without refitting.

[StandardScaler learns means and scales from training samples](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html).
[PCA centers but does not itself standardize feature scales](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html).

For each model dataset, the addon fits 10 separate CV preprocessors and one
separate all-development preprocessor, giving 22 fits in total.

| Preprocessor | Rows permitted during fitting | Intended use |
|---|---|---|
| `cv_fold_00.npz` | Development rows with fold other than 0 | Train/validate CV fold 0 |
| Other `cv_fold_XX.npz` files | Development rows excluding that validation fold | Their respective CV runs |
| `all_development.npz` | All development rows, without holdout or Na/K | Final development-trained model |

Do not reuse `all_development.npz` inside cross-validation. The supplied
`prepare_fold` helper rejects that misuse and checks the exact training-row
identities and feature matrix against the cached CV transform.

## Measured PCA results and what they mean

| Dataset | Raw features | Active after development constant removal | PCA components | Development feature variance retained |
|---|---:|---:|---:|---:|
| Mixed ions | 740 | 497 | 80 | 43.15% |
| Li only | 740 | 431 | 80 | 51.53% |

Across CV training folds, retention was approximately 43.37-44.22% for mixed
ions and 52.07-53.48% for Li only. These percentages describe standardized input
variance. They are not accuracy, target R-squared, or evidence of a particular
voltage error.

The feature set has many sparse categorical indicators; standardizing rare
categories distributes variance across many dimensions. Eighty components
therefore discard a substantial amount of input variation. It would be
premature to make PCA-80 the only training configuration.

The next training step should compare PCA-80 with other component counts and a
scaled representation without PCA using the fixed validation folds. No holdout
performance has been inspected to choose a representation. All raw features
remain available, and `Preprocessor.transform_scaled(X)` supports the no-PCA
branch using the same training-only scaler.

## Files created

Under `data\features\<step5_run_id>\mixed_ions\` and `li_only\`:

| File | Contents |
|---|---|
| `development.npz` | Raw `X`, separate `y`, row/group IDs, working ions and CV folds |
| `holdout.npz` | Raw holdout inputs and separate labels |
| `held_out_Na.npz` | Reserved sodium transfer inputs and labels |
| `held_out_K.npz` | Reserved potassium transfer inputs and labels |
| `preprocessors\cv_fold_00.npz` through `cv_fold_09.npz` | Fold-specific numeric preprocessing states |
| `preprocessors\all_development.npz` | Separate full-development preprocessing state |

Both model datasets contain their own aligned Na/K feature files for evaluation
against their respective preprocessors. These are the same held-out rows, not
additional samples. Their Step 4 known-group/new-group transfer flags can be
joined from the original split CSVs using `row_uid`.

Under `reports\<step5_run_id>\`:

- `step5_report.json`: measured shapes, fit scopes, variance ratios, support diagnostics.
- `step5_report.md`: readable report.
- `feature_schema.json`: complete ordered feature definitions and schema hash.
- `step5_manifest.json`: source-run identity and hashes of output files.

NPZ files are compressed NumPy containers, not Excel files. The loaders use
`allow_pickle=False`. Saved preprocessing state contains numeric arrays and JSON
metadata, without pickled sklearn objects. Each transform was compared with
the corresponding sklearn result before saving.

Every archive is written to a temporary file, reopened and compared with all
original arrays before it is published at the final path. The final reference
run's eight feature archives and 22 preprocessing archives were also loaded
again successfully during the output checks.

Loading an example CV dataset for the next step:

```python
from pathlib import Path
from src.features.preprocessing import FeatureData, Preprocessor, prepare_fold

# Replace this placeholder with the actual Step 5 run folder printed locally.
base = Path('data/features/<step5_run_id>/mixed_ions')
data = FeatureData.load(base / 'development.npz')
pp = Preprocessor.load(base / 'preprocessors/cv_fold_00.npz')
X_train, y_train, X_valid, y_valid = prepare_fold(data, fold=0, saved_preprocessor=pp)
```

## Transfer and unresolved data limitations

The working ion is constant in Li-only development. Direct working-ion identity,
atomic-number and period features therefore cannot teach that model how voltage
changes between working ions. They are removed as training-constant columns.
Endpoint composition features still change for Na/K, but zero-shot performance
must be measured rather than assumed.

The report records rows whose feature values change in columns that were
constant during training. In the reference run, this affected 12 mixed-ion
holdout rows, 11 Li-only holdout rows, and all Na/K rows for both models. Such
features are not learned by the projection. This diagnostic is not an uncertainty
interval or a statement that every affected prediction is unusable.

The representation does not include atomic coordinates, ionic radii or
electronegativity descriptors. It cannot distinguish every structure sharing
the same composition and symmetry summary. It is a transparent baseline for
subsequent validation, not a claim of novel or optimal materials representation.

The 14 priority voltage-review cases retain their original labels. Source DFT
energies, reference-metal energies and calculation settings were not supplied,
so their physical validity remains unresolved. Source database-version
continuity after recovery remains unverified.

## Verification and troubleshooting

Seventeen synthetic tests passed. They check known composition fractions,
training/prediction feature equivalence, independence from targets and source
metadata, rejection of bad inputs, isolation from validation data, group
boundaries, saved-transform matching, and numeric save/load equivalence.

The actual data completed all 22 preprocessing fits. No DNN, SVR or KRR model
was fitted, and no prediction metrics were computed.

Optional local tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_step5_features.py" -v
```

| Error | Action |
|---|---|
| `No module named src.features` | Merge the addon `src` folder into the same existing project. |
| `No module named src.data.chemistry` | The Step 4 source modules are missing; restore the Step 4 addon. |
| Step 4 manifest not found | Use the exact Step 4 run name printed in your successful output. |
| Step 4 integrity check failed | Preserve the edited files and share the message; Step 5 does not silently use changed splits. |
| Incomplete Step 5 output exists | Preserve the folder and share the error for recovery. |
| Not enough training rows or varying columns | Share the error; the code does not pad missing PCA dimensions. |
| Feature schema mismatch | Use a matching feature file, source version and preprocessor. |
| Previous Step 5 output changed or missing | Preserve the files; reruns verify completed outputs rather than overwrite edits. |

## Stop after Step 5

Share the final console output or `step5_report.json`. We will then prepare
Step 6 to train and compare DNN, SVR and KRR using the saved validation folds,
including the representation comparisons justified by the PCA report.
