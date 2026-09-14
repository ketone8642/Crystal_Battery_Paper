"""Step 4: composition validation, provenance checks, and fixed grouped splits.

No labels are corrected or clipped. Voltage flags do not determine eligibility
or split assignment. Na/K remain held-out ions from the same database.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from src.data.chemistry import composition, host_and_loading, host_key, reduced


CORE_IONS = ('Li', 'Mg', 'Ca', 'Zn', 'Al', 'Y')
TRANSFER_IONS = ('Na', 'K')
REQUIRED = (
    'record_key', 'row_type', 'interval_index', 'working_ion', 'framework_formula',
    'formula_charge', 'formula_discharge', 'id_charge', 'id_discharge',
    'fracA_charge', 'fracA_discharge', 'average_voltage_V', 'crystal_system_charge',
    'crystal_system_discharge', 'spacegroup_number_charge', 'spacegroup_number_discharge',
)


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    holdout_fraction: float = 0.10
    cv_folds: int = 10
    fraction_tolerance: float = 1e-6
    voltage_review_low: float = -3.0
    voltage_review_high: float = 10.0
    label_conflict_tolerance: float = 1e-8

    def validate(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError('seed must be an integer')
        if not 0 < self.holdout_fraction < 1:
            raise ValueError('holdout_fraction must be between 0 and 1')
        if not isinstance(self.cv_folds, int) or isinstance(self.cv_folds, bool) or self.cv_folds < 2:
            raise ValueError('cv_folds must be an integer of at least 2')
        if not 0 < self.fraction_tolerance < 0.01:
            raise ValueError('fraction_tolerance must be positive and below 0.01')
        if not (math.isfinite(self.voltage_review_low) and math.isfinite(self.voltage_review_high)
                and self.voltage_review_low < self.voltage_review_high):
            raise ValueError('invalid voltage review bounds')
        if not 0 <= self.label_conflict_tolerance < 0.01:
            raise ValueError('invalid label conflict tolerance')


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def number(value):
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def crystal_system(number):
    for upper, name in [(2, 'triclinic'), (15, 'monoclinic'), (74, 'orthorhombic'),
                        (142, 'tetragonal'), (167, 'trigonal'), (194, 'hexagonal'), (230, 'cubic')]:
        if 1 <= number <= upper:
            return name
    return None


def check_row(raw, source_row, settings):
    row = dict(raw)
    problems, warnings = [], []
    row['source_csv_row'] = source_row
    row['row_uid'] = stable_hash([raw['record_key'], raw['interval_index']])
    row['input_row_hash'] = stable_hash(raw)
    row['canonical_host'] = ''
    ion = raw['working_ion']
    if ion not in CORE_IONS + TRANSFER_IONS:
        problems.append('unsupported_working_ion')
    if raw['row_type'] != 'adjacent':
        problems.append('row_is_not_an_adjacent_interval')
    if not raw['record_key'].strip():
        problems.append('missing_source_record_key')
    if not raw['interval_index'].isdigit():
        problems.append('invalid_interval_index')
    if raw.get('issues', '').strip():
        problems.append('unresolved_step3_issue')
    if not raw.get('battery_id', '').strip():
        warnings.append('battery_id_absent_using_record_key')
    parsed = {}
    for suffix in ('charge', 'discharge'):
        for name in (f'computed_fracA_{suffix}', f'ion_per_reduced_host_{suffix}', f'fraction_error_{suffix}'):
            row[name] = None
        if not raw['id_' + suffix].strip():
            problems.append('missing_endpoint_id_' + suffix)
        supplied = number(raw['fracA_' + suffix])
        if supplied is None or not 0 <= supplied < 1:
            problems.append('invalid_fraction_' + suffix)
        try:
            host, loading, fraction = host_and_loading(raw['formula_' + suffix], ion)
            parsed[suffix] = host
            row[f'computed_fracA_{suffix}'] = fraction
            row[f'ion_per_reduced_host_{suffix}'] = loading
            if supplied is not None:
                row[f'fraction_error_{suffix}'] = abs(fraction - supplied)
                if abs(fraction - supplied) > settings.fraction_tolerance:
                    problems.append('formula_fraction_mismatch_' + suffix)
        except ValueError:
            problems.append('unsupported_or_invalid_formula_' + suffix)
    try:
        framework = reduced(dict(composition(raw['framework_formula'])))
    except ValueError:
        framework = None
        problems.append('unsupported_or_invalid_framework_formula')
    row['delta_ion_per_reduced_host'] = None
    if len(parsed) == 2:
        if parsed['charge'] != parsed['discharge']:
            problems.append('host_composition_changed')
        elif parsed['charge'] != framework:
            problems.append('framework_composition_mismatch')
        else:
            row['canonical_host'] = host_key(parsed['charge'])
        delta = row['ion_per_reduced_host_discharge'] - row['ion_per_reduced_host_charge']
        row['delta_ion_per_reduced_host'] = delta
        if delta <= 0:
            problems.append('non_increasing_ion_loading')
    row['composition_checks_passed'] = not problems
    complete_symmetry = True
    for suffix in ('charge', 'discharge'):
        text = raw['spacegroup_number_' + suffix]
        system = raw['crystal_system_' + suffix].strip().lower()
        if not text or not system:
            complete_symmetry = False
            problems.append('missing_symmetry_' + suffix)
        elif not text.isdigit() or not 1 <= int(text) <= 230:
            complete_symmetry = False
            problems.append('invalid_spacegroup_' + suffix)
        elif crystal_system(int(text)) != system:
            complete_symmetry = False
            problems.append('spacegroup_crystal_system_mismatch_' + suffix)
    row['symmetry_checks_passed'] = complete_symmetry
    voltage = number(raw['average_voltage_V'])
    if voltage is None:
        problems.append('invalid_voltage_label')
    elif voltage < settings.voltage_review_low or voltage > settings.voltage_review_high:
        warnings.append('priority_voltage_review')
    if voltage is not None and voltage < 0:
        warnings.append('negative_voltage')
    for flag in raw.get('warnings', '').split(';'):
        if flag.startswith('source:') or flag.startswith('deprecated_material_'):
            warnings.append(flag)
    row['_problems'] = problems
    row['review_flags'] = ';'.join(sorted(set(warnings)))
    row['priority_voltage_review'] = 'priority_voltage_review' in warnings
    return row


def validate_rows(raw_rows, settings):
    settings.validate()
    if not raw_rows:
        raise ValueError('The input CSV is empty')
    missing = set(REQUIRED) - set(raw_rows[0])
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    if any(not isinstance(r.get(k), str) for r in raw_rows for k in REQUIRED):
        raise ValueError('Input rows have missing cells or an invalid CSV structure')
    rows = [check_row(raw, i + 2, settings) for i, raw in enumerate(raw_rows)]
    # A shared endpoint ID must not describe different compositions or symmetry.
    endpoint_rows, endpoint_values = defaultdict(set), defaultdict(lambda: defaultdict(set))
    for i, row in enumerate(rows):
        for side in ('charge', 'discharge'):
            mid = row['id_' + side]
            if not mid:
                continue
            endpoint_rows[mid].add(i)
            try:
                endpoint_values[mid]['formula'].add(reduced(dict(composition(row['formula_' + side]))))
            except ValueError:
                pass
            sg, system = row['spacegroup_number_' + side], row['crystal_system_' + side]
            if sg and system:
                endpoint_values[mid]['symmetry'].add((sg, system.lower()))
    bad_ids = {mid for mid, info in endpoint_values.items() if any(len(v) > 1 for v in info.values())}
    for mid in bad_ids:
        for i in endpoint_rows[mid]:
            rows[i]['_problems'].append('conflicting_endpoint_metadata')
    # Conservatively collapse only fully identical raw rows. Other repeated
    # endpoint reactions require review, even when voltage labels agree.
    exact_rows, reaction_rows, source_rows = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        exact_rows[row['input_row_hash']].append(i)
        source_rows[row['row_uid']].append(i)
        if row['id_charge'] and row['id_discharge']:
            reaction_rows[(row['working_ion'], row['id_charge'], row['id_discharge'])].append(i)
    duplicates = set()
    for members in exact_rows.values():
        for i in members[1:]:
            rows[i]['_problems'].append('exact_duplicate_input_row')
            duplicates.add(i)
    for members in source_rows.values():
        if len({rows[i]['input_row_hash'] for i in members}) > 1:
            for i in members:
                rows[i]['_problems'].append('conflicting_source_interval_identity')
    repeated, conflicts = 0, 0
    for members in reaction_rows.values():
        members = [i for i in members if i not in duplicates]
        if len(members) <= 1:
            continue
        repeated += 1
        labels = [number(rows[i]['average_voltage_V']) for i in members]
        conflict = any(v is None for v in labels) or max(labels) - min(labels) > settings.label_conflict_tolerance
        conflicts += int(conflict)
        for i in members:
            rows[i]['_problems'].append('conflicting_reaction_labels' if conflict else 'repeated_reaction_requires_provenance_review')
    for row in rows:
        row['step4_issues'] = ';'.join(sorted(set(row.pop('_problems'))))
        row['eligible_for_baseline'] = not row['step4_issues']
    return rows, {
        'conflicting_endpoint_metadata_ids': sorted(bad_ids),
        'exact_duplicate_rows_excluded': len(duplicates),
        'repeated_endpoint_reaction_groups': repeated,
        'conflicting_endpoint_reaction_label_groups': conflicts,
    }


def group_rows(rows):
    """Connected components across all rows, including rows lacking symmetry."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [('record', row['record_key']), ('host', row['canonical_host']),
                ('endpoint', row['id_charge']), ('endpoint', row['id_discharge'])]
        for kind, value in keys:
            if not value:
                continue
            token = (kind, value)
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row)
    for component in members.values():
        gid = 'g_' + min(row['row_uid'] for row in component)
        for row in component:
            row['group_id'] = gid
    return {'connected_groups_all_rows': len(members), 'largest_group_rows': max(map(len, members.values()))}


def fold_assignment(rows, settings, salt):
    sizes = Counter(r['group_id'] for r in rows)
    if len(sizes) < settings.cv_folds:
        raise ValueError(f'{salt}: fewer development groups than requested CV folds')
    # Greedy load balancing by group size, with stable seeded tie breaking.
    ordered = sorted(sizes, key=lambda gid: (-sizes[gid], stable_hash([settings.seed, salt, gid])))
    totals, n_groups = [0] * settings.cv_folds, [0] * settings.cv_folds
    mapping = {}
    for gid in ordered:
        fold = min(range(settings.cv_folds), key=lambda i: (totals[i], n_groups[i], i))
        mapping[gid] = fold
        totals[fold] += sizes[gid]
        n_groups[fold] += 1
    return mapping


def assign_splits(rows, settings):
    eligible = [r for r in rows if r['eligible_for_baseline']]
    core = [r for r in eligible if r['working_ion'] in CORE_IONS]
    groups = sorted({r['group_id'] for r in core}, key=lambda gid: stable_hash([settings.seed, 'holdout', gid]))
    n_test = math.ceil(settings.holdout_fraction * len(groups))
    if not groups or n_test >= len(groups):
        raise ValueError('Not enough core-ion groups for development and holdout')
    held = set(groups[:n_test])
    dev = [r for r in core if r['group_id'] not in held]
    li_dev = [r for r in dev if r['working_ion'] == 'Li']
    mixed_folds = fold_assignment(dev, settings, 'mixed_cv')
    li_folds = fold_assignment(li_dev, settings, 'li_cv')
    mixed_groups = {r['group_id'] for r in dev}
    li_groups = {r['group_id'] for r in li_dev}
    for row in rows:
        row['mixed_cv_fold'] = None
        row['li_cv_fold'] = None
        row['shares_group_with_mixed_development'] = None
        row['shares_group_with_li_development'] = None
        if not row['eligible_for_baseline']:
            row['partition'] = 'quarantine'
        elif row['working_ion'] in TRANSFER_IONS:
            row['partition'] = 'held_out_' + row['working_ion']
            row['shares_group_with_mixed_development'] = row['group_id'] in mixed_groups
            row['shares_group_with_li_development'] = row['group_id'] in li_groups
        elif row['group_id'] in held:
            row['partition'] = 'holdout'
        else:
            row['partition'] = 'development'
            row['mixed_cv_fold'] = mixed_folds[row['group_id']]
            if row['working_ion'] == 'Li':
                row['li_cv_fold'] = li_folds[row['group_id']]
    return rows


def overlap_check(left, right):
    def tokens(data, columns):
        return {r[c] for r in data for c in columns if r[c]}
    return {name: len(tokens(left, cols) & tokens(right, cols)) for name, cols in (
        ('shared_groups', ['group_id']), ('shared_source_records', ['record_key']),
        ('shared_endpoint_ids', ['id_charge', 'id_discharge']), ('shared_host_compositions', ['canonical_host'])
    )}


def check_splits(rows, settings):
    checks = {}
    for model, ions, fold_col in [('mixed_ions', CORE_IONS, 'mixed_cv_fold'), ('li_only', ('Li',), 'li_cv_fold')]:
        dev = [r for r in rows if r['working_ion'] in ions and r['partition'] == 'development']
        held = [r for r in rows if r['working_ion'] in ions and r['partition'] == 'holdout']
        if not dev or not held:
            raise ValueError(f'{model}: development or holdout is empty')
        outer = overlap_check(dev, held)
        folds = []
        for fold in range(settings.cv_folds):
            train = [r for r in dev if r[fold_col] != fold]
            valid = [r for r in dev if r[fold_col] == fold]
            if not train or not valid:
                raise ValueError(f'{model}: empty CV training/validation partition')
            overlap = overlap_check(train, valid)
            if any(overlap.values()):
                raise ValueError(f'{model}: CV leakage detected')
            folds.append({'fold': fold, 'train_rows': len(train), 'validation_rows': len(valid), **overlap})
        if any(outer.values()):
            raise ValueError(f'{model}: holdout leakage detected')
        checks[model] = {'development_rows': len(dev), 'holdout_rows': len(held),
                         'holdout_overlap': outer, 'cross_validation': folds}
    if any(r['working_ion'] in TRANSFER_IONS and r['partition'] in ('development', 'holdout') for r in rows):
        raise ValueError('A held-out ion entered the core experiment')
    return checks


def build_report(rows, settings, validation, grouping, splits):
    eligible = [r for r in rows if r['eligible_for_baseline']]
    def summary(data):
        return {'rows': len(data), 'rows_by_ion': dict(sorted(Counter(r['working_ion'] for r in data).items())),
                'groups': len({r['group_id'] for r in data})}
    report = {
        'stage': 'step4_validation_and_grouped_splitting', 'settings': asdict(settings),
        'input_rows': len(rows), 'composition_checks_passed': sum(r['composition_checks_passed'] for r in rows),
        'symmetry_checks_passed': sum(r['symmetry_checks_passed'] for r in rows),
        'eligible_rows': len(eligible), 'quarantined_rows': len(rows) - len(eligible),
        'issue_counts': dict(Counter(flag for r in rows for flag in r['step4_issues'].split(';') if flag)),
        'negative_voltage_rows': sum((number(r['average_voltage_V']) or 0) < 0 for r in rows),
        'priority_voltage_review_rows': sum(r['priority_voltage_review'] for r in rows),
        'negative_voltage_eligible_rows': sum((number(r['average_voltage_V']) or 0) < 0 for r in eligible),
        'priority_voltage_review_eligible_rows': sum(r['priority_voltage_review'] for r in eligible),
        'all_voltage_labels_preserved': True, 'models_trained': False,
        'scaling_or_pca_fitted': False, **validation, **grouping,
        'partitions': {name: summary([r for r in rows if r['partition'] == name]) for name in
                       ('development', 'holdout', 'held_out_Na', 'held_out_K', 'quarantine')},
        'split_checks': splits,
        'transfer_group_overlap': {ion: {model: {
            'shares_development_group': sum(r[f'shares_group_with_{model}_development'] is True for r in rows if r['partition'] == 'held_out_' + ion),
            'new_group_relative_to_development': sum(r[f'shares_group_with_{model}_development'] is False for r in rows if r['partition'] == 'held_out_' + ion),
        } for model in ('mixed', 'li')} for ion in TRANSFER_IONS},
        'limitations': [
            'Formula checks establish composition consistency, not oxidation-state feasibility, crystal stability, or label correctness.',
            'Original DFT energies, reference-metal energies, structures and calculation settings were not supplied in this CSV; voltages cannot be independently recomputed here.',
            'Symmetry is checked for presence and space-group/crystal-system consistency, not independently recalculated from structures.',
            'The host key groups identical reduced non-working-ion composition, including polymorphs; it does not capture every chemically similar family.',
            'Na and K are held-out ions from the same Materials Project retrieval, not an independent experimental dataset or the original paper\'s 32-material Na set.',
            'Transfer rows may share hosts/endpoints with development; report known-group and new-group transfer separately.',
            'Raw CSV labels were inspected for data-quality review before splitting; no performance-based filtering or split selection was performed.',
            'Source database version continuity could not be established after metadata recovery.',
            'Priority voltage bounds are descriptive review settings, not physical validity limits or exclusion criteria.',
        ],
    }
    return report


def prepare(raw_rows, settings=Settings()):
    rows, validation = validate_rows(raw_rows, settings)
    grouping = group_rows(rows)
    assign_splits(rows, settings)
    splits = check_splits(rows, settings)
    return rows, build_report(rows, settings, validation, grouping, splits)
