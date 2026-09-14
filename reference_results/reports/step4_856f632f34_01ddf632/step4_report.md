# Step 4 dataset validation and grouped splits

Input file: `voltage_intervals.csv`. Rows: 7,461.
Input SHA-256: `856f632f34bb6cadbab568c7484467ee57bd915af294a07e5971cbc60249f0da`.

## Measured validation results

| Check | Rows |
|---|---:|
| Passed composition and concentration checks | 7,461 |
| Passed symmetry presence and consistency checks | 7,419 |
| Eligible for the baseline under the documented rules | 7,419 |
| Set aside for unresolved input issues | 42 |
| Negative reference voltages in original input | 647 |
| Negative reference voltages retained in eligible data | 642 |
| Priority voltage review rows | 14 |

No reference voltages were changed, clipped, averaged, or imputed. The priority
voltage list is descriptive and does not exclude records or determine split assignment.

## Fixed experiment partitions

| Partition | Intervals | Connected groups |
|---|---:|---:|
| development | 6,140 | 2,604 |
| holdout | 630 | 290 |
| held_out_Na | 472 | 311 |
| held_out_K | 177 | 141 |
| quarantine | 42 | 32 |

Mixed-ion development/holdout use Li, Mg, Ca, Zn, Al and Y.
Li-only development/holdout are subsets of those same partitions.
Na/K are held-out ions from this database, not independent external experimental data.

| Model dataset | Development rows | Holdout rows |
|---|---:|---:|
| mixed_ions | 6,140 | 630 |
| li_only | 3,021 | 324 |

## Validation and grouping rules

The parser checks the integer-stoichiometry notation present in the CSV, including
nested parentheses. It rejects unsupported fractional, hydrate, isotope, charge or
variable notation for review. It recognizes all 118 element symbols.

For both endpoints, remove the working ion and reduce the remaining atom counts
by their greatest common divisor. The two resulting host compositions must agree
with the reduced framework formula. Working-ion atom fractions are recalculated
from the full endpoint atom counts, and loading must increase per reduced host.
This host normalization follows the purpose described in the
[pymatgen battery documentation](https://pymatgen.org/pymatgen.apps.battery.html).

Space-group numbers must lie between 1 and 230 and agree with the supplied crystal
system. Shared material IDs must have consistent compositions and symmetry.
This does not establish atomic arrangement, oxidation-state feasibility, or stability.

Connected components join records sharing a source record key, an endpoint ID,
or an identical reduced non-working-ion composition. Connections through incomplete
rows are retained conservatively. This groups polymorphs of the same host composition
together, but does not capture every chemically similar material family.

There are 3,013 connected groups across all input rows;
the largest contains 121 rows.

Holdout groups are ranked by SHA-256 of the fixed seed, purpose and group ID.
The first ceiling(0.10 × number of eligible core groups) are reserved under default
settings. The fraction applies to groups; row proportions need not be exactly 90/10.
The same group-versus-row distinction is documented for
[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
This implementation uses its own deterministic hash ranking, not that class.

Within development, whole groups are assigned largest-first to the currently smallest
validation fold, with deterministic seeded tie breaking. Mixed-ion and Li-only
models have separate 10-fold assignments. Holdout membership is shared.

All development/holdout and CV training/validation pairs were checked for shared
group IDs, source records, endpoint IDs and canonical host compositions. Every
overlap count is zero. No split seed was chosen using prediction performance.

## Priority voltage review

Review flags use V < -3 or V > 10.
These are engineering review bounds, not universal physical limits. All negative
voltages are also flagged separately. The CSV alone cannot establish whether an
unusual voltage is correct; original energies and calculation provenance are needed.

| Ion | Charged formula | Discharged formula | Reference voltage (V) |
|---|---|---|---:|
| Li | V2(PO4)3 | LiV2(PO4)3 | -7.754751 |
| Li | MgCr7(SO4)12 | LiMgCr7(SO4)12 | -6.934789 |
| Li | Li6Fe7O15 | Li8Fe7O15 | -4.121741 |
| Li | Li3V4(OF3)3 | Li4V4(OF3)3 | 10.178403 |
| Ca | CaC4 | CaC2 | 10.529148 |
| Li | Mn5OF11 | Li3Mn5OF11 | 10.623050 |
| Li | Li2Fe3(BO3)3 | Li5Fe6(BO3)6 | 11.530566 |
| Li | Li(FeO2)3 | Li2(FeO2)3 | 12.296080 |
| Li | V4(OF3)3 | Li3V4(OF3)3 | 14.052197 |
| Mg | Eu6WO12 | Eu6MgWO12 | 14.246550 |
| Mg | Eu2Fe2O5 | Eu4Mg(Fe2O5)2 | 14.704305 |
| Li | LiV2P4(HO8)2 | Li3V2P4(HO8)2 | 15.549063 |
| Li | Fe2O3F | LiFe8(O3F)4 | 16.123656 |
| Li | LiTi2(PO4)3 | Li3Ti2(PO4)3 | 33.065771 |

## Transfer overlap

The following counts are relative to the saved development sets. A new
group means no connected group overlap under our rule, not proof of
complete chemical novelty. Evaluate these scopes separately.

| Held-out ion | Reference development set | Shared group | New group |
|---|---|---:|---:|
| Na | mixed | 343 | 129 |
| Na | li | 325 | 147 |
| K | mixed | 120 | 57 |
| K | li | 107 | 70 |

## Limits and next stage

- Formula checks establish composition consistency, not oxidation-state feasibility, crystal stability, or label correctness.
- Original DFT energies, reference-metal energies, structures and calculation settings were not supplied in this CSV; voltages cannot be independently recomputed here.
- Symmetry is checked for presence and space-group/crystal-system consistency, not independently recalculated from structures.
- The host key groups identical reduced non-working-ion composition, including polymorphs; it does not capture every chemically similar family.
- Na and K are held-out ions from the same Materials Project retrieval, not an independent experimental dataset or the original paper's 32-material Na set.
- Transfer rows may share hosts/endpoints with development; report known-group and new-group transfer separately.
- Raw CSV labels were inspected for data-quality review before splitting; no performance-based filtering or split selection was performed.
- Source database version continuity could not be established after metadata recovery.
- Priority voltage bounds are descriptive review settings, not physical validity limits or exclusion criteria.

The source-reference energies and calculation settings are needed to
resolve questionable labels; the review CSV leaves those decisions pending.
Before training, define a feature whitelist. Exclude all IDs, row hashes,
partition/fold columns, quality flags, source warnings, number of voltage
steps and the target itself from predictors. Some flags directly use the target.
Fit scaling, PCA and any learned preprocessing only on each training fold.
Keep the reserved holdout and Na/K labels out of model selection.

No features were fitted and no prediction model was trained in Step 4.
