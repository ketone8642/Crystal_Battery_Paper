# Step 3 data dictionary

The CSVs are inspection tables, not a final feature matrix. Empty cells mean
missing or invalid data as described by the `issues` and `warnings` columns.
Raw values are preserved in the JSONL snapshot for auditing.

| Column | Meaning |
| --- | --- |
| `record_key` | SHA-256 hash of the projected parent electrode document. It is a provenance key, not a model feature or a guarantee that unrelated hashes represent unrelated materials. |
| `battery_id` | Identifier supplied by the source for an electrode. |
| `row_type` | `overall` for a whole electrode range, `adjacent` for a provided individual voltage interval. |
| `interval_index` | Zero-based position in the source `adj_pairs` list; blank for overall rows. Use with `record_key` and `row_type` to locate a row in its source. |
| `working_ion` | Inserted/extracted element: Li, Na, K, Mg, Ca, Zn, Al, or Y. This is an element symbol, not the ionic charge. |
| `paper_training_ion` | Whether the ion belongs to the paper's main six-ion selection. This does not assign a training split. |
| `framework_formula` | Parent electrode's host formula supplied by the source. Its consistency with the row endpoints is not chemically checked in Step 3. |
| `formula_charge` | Formula at the charged, lower-working-ion endpoint. |
| `formula_discharge` | Formula at the discharged, higher-working-ion endpoint. |
| `id_charge`, `id_discharge` | Source material IDs for the two endpoints. |
| `fracA_charge`, `fracA_discharge` | Atomic fraction of the working ion at each endpoint, as supplied by MP. These are not directly the insertion coefficient x in a formula such as Li_xFePO4. |
| `average_voltage_V` | Source-computed average voltage for this row's interval, in volts relative to the working-ion metal reference. This is the intended target label, not an input feature. |
| `crystal_system_charge`, `crystal_system_discharge` | Crystal-system labels retrieved separately for each endpoint. |
| `spacegroup_number_charge`, `spacegroup_number_discharge` | Endpoint space-group numbers, valid integers from 1 to 230. A number identifies a symmetry group; it is not a physical magnitude. |
| `num_steps` | Number of steps reported for the parent electrode, repeated as provenance in interval rows. |
| `last_updated` | Parent electrode update value returned by the API. Snapshot retrieval time is stored separately in the manifest/report. |
| `basic_checks_passed` | Supported ion, nonempty endpoint formulas, valid increasing fractions, and finite numeric voltage; also false when an interval has conflicting labels for the same ion and endpoints. |
| `complete_basic_inputs` | Basic checks pass, both endpoint IDs exist, and both endpoints have crystal-system and valid space-group metadata. This does not mean research-ready. |
| `issues` | Semicolon-separated problems that fail basic checks. |
| `warnings` | Missing metadata, source warnings, deprecated materials, or repeated interval identities requiring review. |

## Example: atomic fraction versus insertion coefficient

In LiFePO4 there is one Li atom among seven total atoms: Li + Fe + P + 4 O.
Its lithium atomic fraction is therefore 1/7, approximately 0.142857. In the
notation Li_xFePO4, the insertion coefficient is x = 1. These are different
quantities. FePO4 has lithium fraction zero. The collector preserves MP's
fractions; it does not silently rename them to x.

## Example: overall versus adjacent voltage

Imagine a host with stable compositions at x = 0, x = 0.5, and x = 1. The
interval 0 to 0.5 can have a different voltage from 0.5 to 1. An overall
electrode record also describes the full range 0 to 1. The source can thus
provide two adjacent voltages and one overall voltage describing related
chemistry. They should not become three independent randomly split samples.
An overall voltage generally reflects transferred-charge weighting, not an
unconditionally valid arithmetic mean of interval voltages. This code reads
the supplied labels and does not recalculate them.

## Checks deliberately deferred

- Parse formulas and verify that both endpoints share a consistent host.
- Check the agreement between chemical compositions and supplied fractions.
- Resolve repeated endpoint identities and their calculation provenance.
- Inspect deprecated-material and source warnings before choosing usable rows.
- Select a consistent representation of crystal symmetry for modeling.
- Define material-family groups so related reactions cannot leak across splits.
- Confirm which descriptors can be obtained without using target-derived data.
- Fit preprocessing, feature selection, scaling, and PCA using training data only.

No feature list should be inferred merely from all available CSV columns.
Identifiers, row indexes, hashes, timestamps, status flags, and voltage labels
must not be treated as chemical input features. In particular, do not add
voltage-derived energy-density quantities as predictors of that same voltage.

The paper's exact historical dataset and complete supplementary descriptor
list are not recreated by this download step. Any later replacement descriptor
set must be labeled as an implementation choice rather than an exact copy of
the paper's features.
