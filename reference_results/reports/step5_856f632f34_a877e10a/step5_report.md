# Step 5 feature preparation results

This is a measured preprocessing report, not a prediction-performance report.

Feature schema: `reaction-composition-symmetry-v1`. Raw features: 740.
The exact paper feature specification is unavailable; this representation is a documented adaptation.

| Feature block | Columns |
|---|---:|
| endpoint_element_fractions | 236 |
| endpoint_composition_summaries | 14 |
| working_ion | 10 |
| concentration_interval | 6 |
| crystal_system | 14 |
| spacegroup | 460 |

| Model | Development | Holdout | Active features in development | PCA variance retained |
|---|---:|---:|---:|---:|
| mixed_ions | 6140 | 630 | 497 | 43.15% |
| li_only | 3021 | 324 | 431 | 51.53% |

Variance retention is measured after constant-column removal and standardization.
It is not accuracy, explained target variation, or evidence that PCA improves prediction.
Retain the raw features and compare no-PCA, 80-component and other training-selected
representations using the saved validation folds before evaluating holdout.

Each CV scaler/PCA sees only the nine training folds. The separate development
scaler/PCA sees only the full development partition. Holdout and Na/K feature
matrices are transformed only; their values and labels are not used to fit preprocessing.

Saved scalers and PCA components use numeric NPZ state, without pickled sklearn
objects. The portable transform was compared with sklearn during every fit.

The categorical vocabulary is fixed before reading the data: 118 elements,
eight working ions, seven crystal systems, and 230 space groups for each endpoint.
Atomic numbers follow the
[IUPAC periodic-table order](https://iupac.org/what-we-do/periodic-table-of-elements/).

Standardization learns training means and scales as described by
[StandardScaler](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html).
Full-SVD PCA is applied without whitening; PCA itself centers inputs but does not scale features,
as described in the [PCA documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html).

## Limits retained from the data and representation

- The fixed 740-column representation is an adaptation, not the unavailable original 237-feature implementation.
- Composition and symmetry do not specify atomic coordinates or independently validate voltage labels.
- The 14 priority voltage-review records retain their original labels; source-energy verification remains pending.
- Atomic-number summaries are simple descriptors; electronegativity, ionic radii and learned crystal representations are not included.
- Standardizing rare categorical columns can spread variance over many components. PCA dimensionality must be assessed through validation, not assumed adequate.
- The working ion never varies within Li-only training. Its direct ion-identity columns are training-constant and removed; zero-shot Na/K performance is unverified.
- Changed training-constant features are reported on held-out data and are not learned by the fitted projection. Such diagnostics are not calibrated prediction uncertainty.
- Na/K are held-out ions from the same database; known-group and new-group transfer must be evaluated separately.
- Numerical fits can differ slightly across Python, NumPy, sklearn and BLAS versions. Each model must use its own saved matching preprocessor.

No neural network, SVR or KRR was fitted. No MAE, RMSE or R-squared is claimed.
