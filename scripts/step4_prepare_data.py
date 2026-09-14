"""Validate the Step 3 interval CSV and save grouped development/holdout/CV splits.

Run from PowerShell with the existing .venv Python. No API key or installation
is needed. This step does not fit features, PCA, or prediction models.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preparation import CORE_IONS, Settings, prepare, stable_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(report, rows):
    lines = [
        '# Step 4 dataset validation and grouped splits', '',
        f"Input file: `{report['input_file']}`. Rows: {report['input_rows']:,}.",
        f"Input SHA-256: `{report['input_sha256']}`.", '',
        '## Measured validation results', '',
        '| Check | Rows |', '|---|---:|',
        f"| Passed composition and concentration checks | {report['composition_checks_passed']:,} |",
        f"| Passed symmetry presence and consistency checks | {report['symmetry_checks_passed']:,} |",
        f"| Eligible for the baseline under the documented rules | {report['eligible_rows']:,} |",
        f"| Set aside for unresolved input issues | {report['quarantined_rows']:,} |",
        f"| Negative reference voltages in original input | {report['negative_voltage_rows']:,} |",
        f"| Negative reference voltages retained in eligible data | {report['negative_voltage_eligible_rows']:,} |",
        f"| Priority voltage review rows | {report['priority_voltage_review_rows']:,} |", '',
        'No reference voltages were changed, clipped, averaged, or imputed. The priority',
        'voltage list is descriptive and does not exclude records or determine split assignment.', '',
        '## Fixed experiment partitions', '',
        '| Partition | Intervals | Connected groups |', '|---|---:|---:|',
    ]
    for name, summary in report['partitions'].items():
        lines.append(f"| {name} | {summary['rows']:,} | {summary['groups']:,} |")
    lines.extend(['', 'Mixed-ion development/holdout use Li, Mg, Ca, Zn, Al and Y.',
                  'Li-only development/holdout are subsets of those same partitions.',
                  'Na/K are held-out ions from this database, not independent external experimental data.', '',
                  '| Model dataset | Development rows | Holdout rows |', '|---|---:|---:|'])
    for name, check in report['split_checks'].items():
        lines.append(f"| {name} | {check['development_rows']:,} | {check['holdout_rows']:,} |")
    lines.extend(['', '## Validation and grouping rules', '',
        'The parser checks the integer-stoichiometry notation present in the CSV, including',
        'nested parentheses. It rejects unsupported fractional, hydrate, isotope, charge or',
        'variable notation for review. It recognizes all 118 element symbols.', '',
        'For both endpoints, remove the working ion and reduce the remaining atom counts',
        'by their greatest common divisor. The two resulting host compositions must agree',
        'with the reduced framework formula. Working-ion atom fractions are recalculated',
        'from the full endpoint atom counts, and loading must increase per reduced host.',
        'This host normalization follows the purpose described in the',
        '[pymatgen battery documentation](https://pymatgen.org/pymatgen.apps.battery.html).', '',
        'Space-group numbers must lie between 1 and 230 and agree with the supplied crystal',
        'system. Shared material IDs must have consistent compositions and symmetry.',
        'This does not establish atomic arrangement, oxidation-state feasibility, or stability.', '',
        'Connected components join records sharing a source record key, an endpoint ID,',
        'or an identical reduced non-working-ion composition. Connections through incomplete',
        'rows are retained conservatively. This groups polymorphs of the same host composition',
        'together, but does not capture every chemically similar material family.', '',
        f"There are {report['connected_groups_all_rows']:,} connected groups across all input rows;",
        f"the largest contains {report['largest_group_rows']} rows.", '',
        'Holdout groups are ranked by SHA-256 of the fixed seed, purpose and group ID.',
        'The first ceiling(0.10 × number of eligible core groups) are reserved under default',
        'settings. The fraction applies to groups; row proportions need not be exactly 90/10.',
        'The same group-versus-row distinction is documented for',
        '[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).',
        'This implementation uses its own deterministic hash ranking, not that class.', '',
        'Within development, whole groups are assigned largest-first to the currently smallest',
        'validation fold, with deterministic seeded tie breaking. Mixed-ion and Li-only',
        'models have separate 10-fold assignments. Holdout membership is shared.', '',
        'All development/holdout and CV training/validation pairs were checked for shared',
        'group IDs, source records, endpoint IDs and canonical host compositions. Every',
        'overlap count is zero. No split seed was chosen using prediction performance.', '',
        '## Priority voltage review', '',
        f"Review flags use V < {report['settings']['voltage_review_low']:g} or V > {report['settings']['voltage_review_high']:g}.",
        'These are engineering review bounds, not universal physical limits. All negative',
        'voltages are also flagged separately. The CSV alone cannot establish whether an',
        'unusual voltage is correct; original energies and calculation provenance are needed.', '',
        '| Ion | Charged formula | Discharged formula | Reference voltage (V) |', '|---|---|---|---:|',
    ])
    for row in sorted((r for r in rows if r['priority_voltage_review']), key=lambda r: float(r['average_voltage_V'])):
        lines.append(f"| {row['working_ion']} | {row['formula_charge']} | {row['formula_discharge']} | {float(row['average_voltage_V']):.6f} |")
    lines.extend(['', '## Transfer overlap', '',
                  'The following counts are relative to the saved development sets. A new',
                  'group means no connected group overlap under our rule, not proof of',
                  'complete chemical novelty. Evaluate these scopes separately.', '',
                  '| Held-out ion | Reference development set | Shared group | New group |', '|---|---|---:|---:|'])
    for ion, models in report['transfer_group_overlap'].items():
        for model, counts in models.items():
            lines.append(f"| {ion} | {model} | {counts['shares_development_group']} | {counts['new_group_relative_to_development']} |")
    lines.extend(['', '## Limits and next stage', ''])
    lines.extend('- ' + item for item in report['limitations'])
    lines.extend(['', 'The source-reference energies and calculation settings are needed to',
                  'resolve questionable labels; the review CSV leaves those decisions pending.',
                  'Before training, define a feature whitelist. Exclude all IDs, row hashes,',
                  'partition/fold columns, quality flags, source warnings, number of voltage',
                  'steps and the target itself from predictors. Some flags directly use the target.',
                  'Fit scaling, PCA and any learned preprocessing only on each training fold.',
                  'Keep the reserved holdout and Na/K labels out of model selection.', '',
                  'No features were fitted and no prediction model was trained in Step 4.'])
    return '\n'.join(lines) + '\n'


def show_summary(report, report_dir):
    print(f"Input intervals: {report['input_rows']}")
    print(f"Composition checks passed: {report['composition_checks_passed']}")
    print(f"Eligible intervals: {report['eligible_rows']}")
    print(f"Quarantined intervals: {report['quarantined_rows']}")
    for name, check in report['split_checks'].items():
        print(f"{name}: development={check['development_rows']}, holdout={check['holdout_rows']}")
    for ion in ('Na', 'K'):
        print(f"Held-out {ion} intervals: {report['partitions']['held_out_' + ion]['rows']}")
    print('Group, source record, endpoint and host overlap: 0 in holdout and all CV checks')
    print('STEP 4 VALIDATION AND SPLITTING COMPLETE')
    print(f'Report: {report_dir / "step4_report.json"}')
    print('No model has been trained. Review these results before Step 5.')


def run(input_path, project_root, settings=Settings()):
    settings.validate()
    input_path = input_path.resolve()
    input_hash = sha(input_path)
    config = {'algorithm_version': 'step4-v1', **asdict(settings)}
    run_id = 'step4_' + input_hash[:10] + '_' + stable_hash(config)[:8]
    processed = project_root / 'data' / 'processed' / run_id
    split_dir = project_root / 'data' / 'splits' / run_id
    report_dir = project_root / 'reports' / run_id
    manifest_path = report_dir / 'step4_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'complete':
            raise ValueError(f'An incomplete output run exists at {report_dir}. Preserve it and share the error before retrying.')
        for rel, expected in manifest['output_sha256'].items():
            path = project_root / rel
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f'A previous output was changed or is missing: {rel}. Existing results were not overwritten.')
        report = json.loads((report_dir / 'step4_report.json').read_text(encoding='utf-8'))
        print('Verified existing results for this input and configuration.')
        show_summary(report, report_dir)
        return report
    if any(path.exists() for path in (processed, split_dir, report_dir)):
        raise ValueError('An output directory already exists without a complete manifest; it was not overwritten.')
    with input_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        raw_fields = reader.fieldnames
        raw = list(reader)
    if not raw_fields or len(raw_fields) != len(set(raw_fields)) or any(None in row for row in raw):
        raise ValueError('Invalid CSV: missing/duplicate headers or extra cells')
    rows, report = prepare(raw, settings)
    report.update({'run_id': run_id, 'input_file': input_path.name, 'input_sha256': input_hash,
                   'algorithm_version': config['algorithm_version'],
                   'created_at_utc': datetime.now(timezone.utc).isoformat()})
    for path in (processed, split_dir, report_dir):
        path.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'writing', 'input_sha256': input_hash, 'configuration': config}
    save_json(manifest_path, manifest)
    outputs = []
    try:
        fields = raw_fields + [key for key in rows[0] if key not in raw_fields]
        ordered = sorted(rows, key=lambda row: (row['row_uid'], row['source_csv_row']))
        def csv_out(path, data, columns=fields):
            save_csv(path, data, columns)
            outputs.append(path)
        csv_out(processed / 'interval_audit.csv', ordered)
        csv_out(processed / 'validated_intervals.csv', [r for r in ordered if r['eligible_for_baseline']])
        csv_out(processed / 'quarantine_intervals.csv', [r for r in ordered if not r['eligible_for_baseline']])
        priority = [dict(r, review_decision='pending', evidence_reference='', reviewer_notes='')
                    for r in ordered if r['priority_voltage_review']]
        csv_out(processed / 'priority_voltage_review.csv', priority,
                fields + ['review_decision', 'evidence_reference', 'reviewer_notes'])
        for model, ions in [('mixed_ions', CORE_IONS), ('li_only', ('Li',))]:
            folder = split_dir / model
            folder.mkdir()
            for partition in ('development', 'holdout'):
                csv_out(folder / (partition + '.csv'), [r for r in ordered if r['working_ion'] in ions and r['partition'] == partition])
        transfer = split_dir / 'held_out_ions'
        transfer.mkdir()
        for ion in ('Na', 'K'):
            csv_out(transfer / (ion + '.csv'), [r for r in ordered if r['partition'] == 'held_out_' + ion])
        assignment_fields = ['row_uid', 'record_key', 'interval_index', 'working_ion', 'canonical_host',
                             'group_id', 'eligible_for_baseline', 'partition', 'mixed_cv_fold', 'li_cv_fold']
        csv_out(split_dir / 'split_assignments.csv', ordered, assignment_fields)
        json_path = report_dir / 'step4_report.json'
        save_json(json_path, report)
        outputs.append(json_path)
        md_path = report_dir / 'step4_report.md'
        md_path.write_text(report_markdown(report, rows), encoding='utf-8')
        outputs.append(md_path)
        manifest.update({'status': 'complete', 'output_sha256': {
            str(path.relative_to(project_root)).replace('\\', '/'): sha(path) for path in outputs
        }})
        save_json(manifest_path, manifest)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['failure_type'] = type(exc).__name__
        save_json(manifest_path, manifest)
        raise
    show_summary(report, report_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Step 3 voltage_intervals.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout-fraction', type=float, default=0.10)
    parser.add_argument('--cv-folds', type=int, default=10)
    args = parser.parse_args()
    try:
        run(args.input, ROOT, Settings(seed=args.seed, holdout_fraction=args.holdout_fraction, cv_folds=args.cv_folds))
        return 0
    except (ValueError, OSError) as exc:
        print(f'[ERROR] {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
