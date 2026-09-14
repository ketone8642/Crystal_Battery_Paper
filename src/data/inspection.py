"""Produce auditable tables; chemistry validation and train/test splitting follow later."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

from src.data.mp_download import element_symbol, fingerprint


ROW_FIELDS = [
    "record_key", "battery_id", "row_type", "interval_index", "working_ion",
    "paper_training_ion", "framework_formula", "formula_charge", "formula_discharge",
    "id_charge", "id_discharge", "fracA_charge", "fracA_discharge",
    "average_voltage_V", "crystal_system_charge", "spacegroup_number_charge",
    "crystal_system_discharge", "spacegroup_number_discharge", "num_steps",
    "last_updated", "basic_checks_passed", "complete_basic_inputs", "issues", "warnings",
]


def valid_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def valid_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def make_row(parent: dict, pair: dict, row_type: str, interval_index: int | None, material_map: dict) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    ion = element_symbol(parent.get("working_ion"))
    if ion is None:
        issues.append("missing_or_unsupported_working_ion")
    row = {name: None for name in ROW_FIELDS}
    row.update({
        "record_key": fingerprint(parent),
        "battery_id": parent.get("battery_id"),
        "row_type": row_type,
        "interval_index": interval_index,
        "working_ion": ion,
        "paper_training_ion": ion in {"Li", "Mg", "Ca", "Zn", "Al", "Y"},
        "framework_formula": parent.get("framework_formula"),
        "num_steps": parent.get("num_steps"),
        "last_updated": parent.get("last_updated"),
    })
    if not valid_text(row["battery_id"]):
        warnings.append("missing_battery_id_use_record_key_for_provenance")
    for suffix in ("charge", "discharge"):
        for field in (f"formula_{suffix}", f"id_{suffix}", f"fracA_{suffix}"):
            # Never fill interval endpoints/labels using the parent's full range.
            row[field] = pair.get(field)
        if not valid_text(row[f"formula_{suffix}"]):
            issues.append(f"missing_formula_{suffix}")
        fraction = row[f"fracA_{suffix}"]
        if not valid_number(fraction) or not 0 <= fraction < 1:
            issues.append(f"invalid_fracA_{suffix}")
        material_id = row[f"id_{suffix}"]
        if not valid_text(material_id):
            warnings.append(f"missing_id_{suffix}")
            material = {}
        else:
            material = material_map.get(material_id, {})
            if not material:
                warnings.append(f"missing_material_metadata_{suffix}")
        symmetry = material.get("symmetry") or {}
        if not isinstance(symmetry, dict):
            symmetry = {}
        crystal_system = symmetry.get("crystal_system")
        number = symmetry.get("number")
        if valid_text(crystal_system):
            row[f"crystal_system_{suffix}"] = crystal_system
        else:
            warnings.append(f"missing_crystal_system_{suffix}")
        if isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= 230:
            row[f"spacegroup_number_{suffix}"] = number
        else:
            warnings.append(f"missing_spacegroup_number_{suffix}")
        if material.get("deprecated") is True:
            warnings.append(f"deprecated_material_{suffix}")
    lo, hi = row["fracA_charge"], row["fracA_discharge"]
    if valid_number(lo) and valid_number(hi) and hi <= lo:
        issues.append("non_increasing_working_ion_fraction")
    voltage = pair.get("average_voltage")
    if not valid_number(voltage):
        issues.append("missing_or_nonfinite_voltage")
    else:
        # Negative and zero labels are valid numbers. No voltage-range filtering.
        row["average_voltage_V"] = voltage
    source_warnings = parent.get("warnings") or []
    if isinstance(source_warnings, list):
        warnings.extend(f"source:{str(item)}" for item in source_warnings)
    elif source_warnings:
        warnings.append(f"source:{source_warnings}")
    row["basic_checks_passed"] = not issues
    row["complete_basic_inputs"] = not issues and all(
        row[f"{field}_{suffix}"] is not None
        for field in ("crystal_system", "spacegroup_number")
        for suffix in ("charge", "discharge")
    ) and all(valid_text(row[f"id_{s}"]) for s in ("charge", "discharge"))
    row["issues"] = ";".join(issues)
    row["warnings"] = ";".join(warnings)
    return row


def inspect_documents(documents: list[dict], materials: list[dict]) -> tuple[list[dict], list[dict], dict]:
    material_map = {m["material_id"]: m for m in materials if valid_text(m.get("material_id"))}
    summaries: list[dict] = []
    intervals: list[dict] = []
    seen: set[str] = set()
    duplicate_count = 0
    no_pairs = 0
    malformed_pairs = 0
    for document in documents:
        key = fingerprint(document)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        summaries.append(make_row(document, document, "overall", None, material_map))
        pairs = document.get("adj_pairs")
        if not isinstance(pairs, list) or not pairs:
            no_pairs += 1
            continue
        for index, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                malformed_pairs += 1
                continue
            intervals.append(make_row(document, pair, "adjacent", index, material_map))

    # Flag repeated interval identities for later resolution instead of silently
    # training on duplicated or conflicting labels. Distinct source records stay
    # visible in the inspection CSVs.
    identity_groups: dict[tuple, list[dict]] = {}
    for row in intervals:
        if valid_text(row["id_charge"]) and valid_text(row["id_discharge"]):
            identity = (row["working_ion"], row["id_charge"], row["id_discharge"])
            identity_groups.setdefault(identity, []).append(row)
    duplicate_interval_groups = 0
    conflicting_interval_groups = 0
    for rows in identity_groups.values():
        if len(rows) > 1:
            duplicate_interval_groups += 1
            labels = [r["average_voltage_V"] for r in rows if valid_number(r["average_voltage_V"])]
            conflict = bool(labels) and max(labels) - min(labels) > 1e-6
            if conflict:
                conflicting_interval_groups += 1
            for row in rows:
                row["warnings"] = ";".join(filter(None, [row["warnings"], "repeated_interval_identity_review_required"]))
                if conflict:
                    row["issues"] = ";".join(filter(None, [row["issues"], "conflicting_interval_labels"]))
                    row["basic_checks_passed"] = False
                    row["complete_basic_inputs"] = False

    def summarize(rows):
        labels = [row["average_voltage_V"] for row in rows if valid_number(row["average_voltage_V"])]
        return {
            "rows": len(rows),
            "rows_by_ion": dict(sorted(Counter(row["working_ion"] or "UNKNOWN" for row in rows).items())),
            "basic_checks_passed": sum(row["basic_checks_passed"] for row in rows),
            "complete_basic_inputs": sum(row["complete_basic_inputs"] for row in rows),
            "missing_or_invalid_voltage": sum(not valid_number(row["average_voltage_V"]) for row in rows),
            "negative_voltage_rows_retained": sum(value < 0 for value in labels),
            "zero_voltage_rows_retained": sum(value == 0 for value in labels),
            "voltage_statistics_V": ({"min": min(labels), "max": max(labels), "mean": statistics.mean(labels), "median": statistics.median(labels)} if labels else {}),
            "issue_counts": dict(Counter(issue for row in rows for issue in row["issues"].split(";") if issue)),
            "warning_counts": dict(Counter(warning for row in rows for warning in row["warnings"].split(";") if warning)),
        }
    report = {
        "label_source": "Materials Project computed insertion voltages, not experimental measurements",
        "reproduction_status": "Current database adaptation; not the original 2019 dataset snapshot",
        "raw_electrode_records": len(documents),
        "exact_duplicate_documents_collapsed": duplicate_count,
        "unique_electrode_documents": len(summaries),
        "documents_without_adjacent_pairs": no_pairs,
        "malformed_adjacent_entries": malformed_pairs,
        "repeated_interval_identity_groups": duplicate_interval_groups,
        "conflicting_interval_label_groups": conflicting_interval_groups,
        "material_metadata_records": len(material_map),
        "overall_summary": summarize(summaries),
        "adjacent_intervals": summarize(intervals),
        "next_stage": "Validate chemical formulas and shared hosts, resolve duplicates and calculation provenance, then define grouped splits before fitting features.",
        "limitations": [
            "Basic checks do not certify chemical validity or suitability as a battery electrode.",
            "Overall and adjacent tables overlap and must not be blindly concatenated.",
            "Space group metadata is retained separately for each endpoint; no structure is guessed.",
            "No target label is imputed, no model is trained, and no train/test split exists yet.",
            "Battery ID alone may not separate related electrodes; shared endpoint/family grouping will be checked later.",
        ],
    }
    return summaries, intervals, report


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Expected an object in {path.name}")
                records.append(item)
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def inspect_snapshot(project_root: Path, snapshot: Path) -> tuple[dict, Path]:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("This snapshot is incomplete. Run the downloader again; incomplete downloads are not exported as completed datasets.")
    documents = read_jsonl(snapshot / "electrodes.jsonl")
    materials = read_jsonl(snapshot / "materials.jsonl")
    if len(documents) != manifest.get("electrode_records") or len(materials) != manifest.get("material_metadata_records"):
        raise ValueError("Snapshot file counts do not match the manifest.")
    hash_files = ["electrodes.jsonl", "materials.jsonl"]
    if "metadata_lookup_issues.jsonl" in manifest.get("sha256", {}):
        hash_files.append("metadata_lookup_issues.jsonl")
    for filename in hash_files:
        import hashlib
        actual = hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()
        if actual != manifest.get("sha256", {}).get(filename):
            raise ValueError(f"Snapshot integrity check failed for {filename}.")
    summaries, intervals, report = inspect_documents(documents, materials)
    report["snapshot"] = snapshot.name
    report["retrieved_at_utc"] = manifest["started_at_utc"]
    report["requested_ions"] = manifest["requested_ions"]
    report["api_reported_database_versions"] = manifest.get("api_reported_database_versions", [])
    report["requested_material_ids"] = manifest.get("requested_material_ids")
    report["missing_material_metadata_ids"] = manifest.get("missing_material_metadata_ids", [])
    report["metadata_lookup_issues_count"] = manifest.get("metadata_issues_count", 0)
    report["metadata_lookup_issues_file"] = (
        "metadata_lookup_issues.jsonl" if (snapshot / "metadata_lookup_issues.jsonl").exists() else None
    )
    if manifest.get("recovery_source_snapshot"):
        report["recovery"] = {key: manifest.get(key) for key in (
            "recovery_source_snapshot", "recovery_started_at_utc", "finished_at_utc",
            "reused_material_metadata_records", "metadata_ids_to_retry",
            "database_version_continuity_verified",
        )}
        report["limitations"].append(
            "Recovered metadata was retrieved after the original electrodes. Cross-session database version continuity is not certified."
        )
    data_dir = project_root / "data" / "processed" / snapshot.name
    report_dir = project_root / "reports" / snapshot.name
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(data_dir / "electrode_summary.csv", summaries)
    write_csv(data_dir / "voltage_intervals.csv", intervals)
    write_csv(data_dir / "intervals_for_review.csv", [r for r in intervals if not r["basic_checks_passed"] or r["warnings"]])
    report_file = report_dir / "dataset_report.json"
    report_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "STEP 3 DATASET INSPECTION",
        f"Snapshot: {snapshot.name}",
        f"Unique electrode documents: {len(summaries)}",
        f"Adjacent voltage intervals: {len(intervals)}",
        f"Intervals passing basic checks: {report['adjacent_intervals']['basic_checks_passed']}",
        f"Intervals with complete basic inputs: {report['adjacent_intervals']['complete_basic_inputs']}",
        f"Endpoint IDs with unresolved symmetry metadata: {len(report['missing_material_metadata_ids'])}",
        "Electrode documents by ion:",
    ]
    lines.extend(f"  {ion}: {count}" for ion, count in report["overall_summary"]["rows_by_ion"].items())
    lines.extend(["", "No model has been trained. Chemical/provenance validation and grouped splitting follow in Step 4."])
    (report_dir / "dataset_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nCSV tables: {data_dir}")
    print(f"Report to share: {report_file}")
    return report, report_file
