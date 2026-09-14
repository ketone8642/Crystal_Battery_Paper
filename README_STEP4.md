# Step 4: validate reaction data and create grouped splits

This addon runs on the existing Windows Python 3.11 project. No API key, package
installation or web server is needed. It validates the uploaded Step 3 CSV,
preserves its original labels and records, and saves the next experiment's
development, holdout and cross-validation assignments. Model training follows
after review of this step.

## Results from the supplied CSV

The included `reference_results` directory contains actual results generated from
your uploaded `voltage_intervals.csv`. These are data-validation results, not
model performance results.

| Check | Result |
|---|---:|
| Input voltage intervals | 7,461 |
| Composition and concentration checks passed | 7,461 |
| Complete symmetry inputs with consistent crystal system and space-group number | 7,419 |
| Rows set aside because symmetry is unavailable | 42 |
| Connected groups across the original rows | 3,013 |
| Negative labels in original CSV | 647 |
| Negative labels retained in the eligible data | 642 |
| Labels above 10 V | 11 |
| Labels below -3 V | 3 |
| Priority voltage review rows | 14 |

All 7,461 reference voltage strings are unchanged. The five negative-voltage rows
outside the eligible set are among the 42 rows with missing symmetry; their
voltage sign did not determine exclusion. The 14 priority review rows are
retained in the eligible data and explicitly flagged. The -3 and 10 V review
bounds are descriptive engineering settings, not proven physical limits.

| Experiment | Development | Reserved holdout |
|---|---:|---:|
| Mixed ions: Li, Mg, Ca, Zn, Al, Y | 6,140 | 630 |
| Li-only subset of the same split | 3,021 | 324 |

Na (472 complete intervals) and K (177 complete intervals) are reserved for
held-out-ion transfer evaluation. They are not included in either model's
development or holdout partition. They come from the same database download,
not an independent experimental source or the original paper's 32-item Na set.

## Install

1. Extract `battery_voltage_step4_validation.zip`.
2. Copy the `scripts`, `src`, `tests` folders and `README_STEP4.md` into your
   existing `battery_voltage_project` folder. Merge folders when prompted.
3. The `reference_results` folder is a ready-to-read copy of the results produced
   here. You may keep it beside the project. The run command below creates the
   working results on your own computer from your own CSV.

Your existing Step 2/3 application and downloaded snapshots remain available.

## Run in PowerShell

Change to the project folder:

```powershell
Set-Location 'D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project'
```

Run Step 4:

```powershell
.\.venv\Scripts\python.exe scripts\step4_prepare_data.py --input 'data\processed\mp_20260913T172046Z_c9d04c_recovered\voltage_intervals.csv'
```

Expected main output for this exact input:

```text
Input intervals: 7461
Composition checks passed: 7461
Eligible intervals: 7419
Quarantined intervals: 42
mixed_ions: development=6140, holdout=630
li_only: development=3021, holdout=324
Held-out Na intervals: 472
Held-out K intervals: 177
Group, source record, endpoint and host overlap: 0 in holdout and all CV checks
STEP 4 VALIDATION AND SPLITTING COMPLETE
```

The exact input produces run ID `step4_856f632f34_01ddf632` under default settings.
Editing or re-saving the input CSV may change its byte hash and therefore the
run ID. Keep the original exported CSV for reproducibility.

## Output files

Under `data\processed\<run_id>\`:

| File | Purpose |
|---|---|
| `interval_audit.csv` | All 7,461 rows, original columns, recomputed checks, flags and assignments |
| `validated_intervals.csv` | 7,419 rows eligible under the documented input rules |
| `quarantine_intervals.csv` | The 42 incomplete rows with explicit reasons |
| `priority_voltage_review.csv` | The 14 flagged voltage rows, with blank evidence/notes and pending review decisions |

Under `data\splits\<run_id>\`:

| Location | Purpose |
|---|---|
| `mixed_ions\development.csv` | Six-ion development set with `mixed_cv_fold` |
| `mixed_ions\holdout.csv` | Reserved six-ion holdout |
| `li_only\development.csv` | Li development subset with `li_cv_fold` |
| `li_only\holdout.csv` | Reserved Li holdout subset |
| `held_out_ions\Na.csv` | Held-out sodium intervals |
| `held_out_ions\K.csv` | Held-out potassium intervals |
| `split_assignments.csv` | Saved row-to-group, partition and fold mapping |

Under `reports\<run_id>\`:

- `step4_report.json`: counts, settings, overlap checks and limitations.
- `step4_report.md`: readable findings and every priority voltage example.
- `step4_manifest.json`: input/configuration identity and output integrity hashes.

An identical rerun verifies and reuses completed outputs. It will not overwrite a
file you have modified. Make a separate working copy of the review CSV if you
wish to enter evidence and decisions.

## What is actually validated

The formula parser supports all 118 element symbols, positive integer counts,
and nested parentheses or brackets, which covers every formula in the supplied
CSV. Unsupported fractional occupancy, hydrate, isotope, charge or variable
notation is rejected for review rather than guessed.

For each interval:

1. Parse both endpoint formulas and the reported framework formula.
2. Remove the working ion and reduce non-working-ion atom counts using their
   greatest common divisor. Both endpoints and the framework must agree.
3. Recalculate the working-ion atom fraction and compare it with `fracA`, using
   an absolute tolerance of 0.000001. Ion loading per reduced host must increase.
4. Check that symmetry information is present, that space-group numbers lie
   between 1 and 230, and that the supplied crystal system matches the number.
5. Check that repeated material IDs have consistent composition and symmetry.
6. Collapse only fully identical raw rows. Quarantine other repeated endpoint
   reactions for provenance review; conflicting labels are never averaged.

All formula fractions in your CSV agree to floating-point precision. This is a
composition-consistency result. It does not validate oxidation states, atomic
geometry, phase stability or the DFT voltage calculations.

The host normalization is motivated by the reference-state explanation in the
[pymatgen battery documentation](https://pymatgen.org/pymatgen.apps.battery.html).
The local parser is limited to this CSV's notation and does not replace
pymatgen for general crystal or chemistry operations.

## Split rules and how to use the folds

Rows belong to one connected group if they share a `record_key`, either endpoint
material ID, or the same reduced non-working-ion host composition. The linkage
is transitive, including paths through incomplete rows. Polymorphs with the same
host composition stay together. This does not cover all chemically related
families; no structure matching was possible from the CSV alone.

Default seed: 42. Holdout fraction: 0.10 of eligible core groups, rounded up.
The remaining groups form development. Groups are ranked by a seeded SHA-256
value, so assignments are deterministic on Windows and Linux without relying
on random-number-library versions. No target values or model scores are used
to select groups. This gives 290 holdout groups and 2,604 development groups.

The fraction applies to groups; the row ratio is therefore approximate. The
same distinction between group and row fractions is explained in
[scikit-learn's GroupShuffleSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
This addon implements its own hash-based partitioner, not GroupShuffleSplit.

Within development, whole groups are placed into 10 validation folds with
greedy row-count balancing. Mixed-ion and Li-only experiments each have their
own fold column. For validation fold `k`:

- Train on development rows whose relevant fold column is not `k`.
- Validate on development rows whose relevant fold column equals `k`.
- Repeat for `k = 0, 1, ..., 9`.

The holdout is never one of these folds. Scaling, PCA and any other learned
preprocessing must be fitted only to the training portion of each fold in the
later training step. After configuration selection, fit on all development
rows and evaluate the reserved holdout once. Retain these assignments when
reviewing later label corrections; do not search for a more favorable split.

## Transfer evaluation needs two scopes

Some Na/K rows share connected groups with development. Their CSV columns
identify this separately for mixed-ion and Li-only models.

| Ion | Reference development | Shared group | New group |
|---|---|---:|---:|
| Na | Mixed ions | 343 | 129 |
| Na | Li only | 325 | 147 |
| K | Mixed ions | 120 | 57 |
| K | Li only | 107 | 70 |

An unfamiliar working ion is different from an unfamiliar host. Report these
transfer scopes separately. A new group under our rule is not proof of complete
chemical novelty. Counts apply to development-trained models; reassess overlap
if a later model is trained on a different population.

## Unresolved voltage provenance

For example, the CSV contains `LiTi2(PO4)3` to `Li3Ti2(PO4)3` at 33.065771 V
(`mp-aaabsuoy` to `mp-aaabrcga`). Its endpoint composition and fraction checks
pass, but those checks do not establish that the voltage is physically reliable.

The uploaded CSV does not include the endpoint total energies, corrections,
working-metal reference energy, or complete calculation settings needed to
independently recompute voltage. All 14 priority review decisions remain pending.
An evidence-backed review should compare the reaction-normalized endpoint
energies, corrections and metal reference using compatible calculations; it
must not invent a replacement voltage from a familiar experimental material.

The original downloader also lacked a persisted database version at failure;
cross-session version continuity after recovery remains unverified.

## Source code and verification

| File | Responsibility |
|---|---|
| `src\data\chemistry.py` | Restricted exact formula parser and host normalization |
| `src\data\preparation.py` | Validation rules, duplicate checks, groups, splits and overlap verification |
| `scripts\step4_prepare_data.py` | PowerShell entry point and CSV/report export |
| `tests\test_step4_data.py` | 17 synthetic regression tests |

All 17 tests passed. The actual uploaded CSV was processed successfully; an
independent output check confirmed preservation of all 7,461 voltage strings,
the 42 incomplete rows, and no holdout overlap. Python 3.11 syntax checks passed.
The Windows CLI itself must still run on your computer.

Optional local test command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_step4_data.py" -v
```

## Common errors

| Error | Action |
|---|---|
| `No module named src.data.preparation` | Copy the addon `src` folder into the same project as the script. |
| Input file not found | Check the recovered snapshot name and use the project directory shown above. |
| Missing CSV columns | Use the original `voltage_intervals.csv`, not `electrode_summary.csv` or an edited export. |
| Not enough groups | Share the error. The code does not silently replace group splitting with row splitting. |
| Prior output changed or missing | Preserve the edited output and share the message; the script will not overwrite it. |
| Incomplete output run exists | Preserve the output folder and share the error for recovery. |
| CSV permission denied | Close the CSV in Excel before running. |

## Stop after Step 4

Share the final console output or `step4_report.json`. The next step will define
the feature representation and its strict predictor whitelist. The original
paper's exact 237-feature implementation has not been recovered, so any feature
replacement must remain documented as an adaptation.

Never feed the model every numeric column in these audit tables. Identifiers,
hashes, partitions, fold labels, review flags, source warnings, number of voltage
steps and voltage-derived fields must not become predictors. No feature fitting,
PCA or model training has been performed by this addon.
