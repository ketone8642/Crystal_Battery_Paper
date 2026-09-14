"""Recover metadata after a verified complete electrode download.

The original snapshot is read-only. Recovery creates a new snapshot, with
source hashes and explicit notes about retrieval-time/version limitations.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from src.data.inspection import read_jsonl
from src.data.mp_download import (
    BASE_URL, ELECTRODE_ROUTE, MATERIAL_ROUTE, IONS, DataAccessError,
    MPClient, element_symbol, fetch_material_batch, fingerprint, requested_material_ids,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_recovery_source(source: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "step3-v1":
        raise DataAccessError("This recovery command requires a Step 3 snapshot.")
    if manifest.get("status") not in {"failed", "interrupted"}:
        raise DataAccessError("Choose the failed or interrupted snapshot shown in the error output.")
    if (manifest.get("source"), manifest.get("electrode_route"), manifest.get("material_route")) != (
        BASE_URL, ELECTRODE_ROUTE, MATERIAL_ROUTE
    ):
        raise DataAccessError("The saved snapshot has unexpected API provenance.")
    documents = read_jsonl(source / "electrodes.jsonl")
    ions = manifest.get("requested_ions")
    if not isinstance(ions, list) or not ions or any(ion not in IONS for ion in ions):
        raise DataAccessError("The saved snapshot has invalid requested ions.")
    counts = Counter(element_symbol(doc.get("working_ion")) for doc in documents)
    expected = manifest.get("electrode_records_by_ion", {})
    if not documents or len(documents) != manifest.get("electrode_records") or set(counts) - set(ions):
        raise DataAccessError("Electrode file counts do not match the saved download manifest.")
    for ion in ions:
        if ion not in expected or expected[ion] != counts[ion]:
            raise DataAccessError(f"The {ion} electrode download was incomplete; metadata-only recovery cannot proceed.")
        pages = [p for p in manifest.get("page_log", []) if p.get("route") == "electrodes" and p.get("working_ion") == ion]
        offset = 0
        for page in pages:
            count = page.get("count")
            if (page.get("skip") != offset or page.get("total") != counts[ion]
                    or isinstance(count, bool) or not isinstance(count, int) or count <= 0):
                raise DataAccessError(f"Cannot verify the saved {ion} electrode page sequence.")
            offset += count
        if offset != counts[ion]:
            raise DataAccessError(f"Cannot verify that all {ion} electrode pages were saved.")
    ids = requested_material_ids(documents)
    if manifest.get("requested_material_ids") != len(ids):
        raise DataAccessError("The source did not finish the electrode stage or its endpoints have changed.")
    for filename, expected_hash in manifest.get("sha256", {}).items():
        if filename in {"electrodes.jsonl", "materials.jsonl", "metadata_lookup_issues.jsonl"}:
            if digest(source / filename) != expected_hash:
                raise DataAccessError(f"Source snapshot integrity check failed for {filename}.")

    wanted = set(ids)
    materials: dict[str, dict] = {}
    conflicts: set[str] = set()
    issues: list[dict] = []
    raw = source / "materials.jsonl"
    saved_count = 0
    if raw.exists():
        lines = raw.read_text(encoding="utf-8").splitlines()
        last_content = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                if i == last_content:
                    issues.append({"kind": "discarded_unfinished_metadata_line", "line_number": i + 1})
                    continue
                raise DataAccessError("The saved metadata contains a damaged line before its end.") from None
            if not isinstance(doc, dict):
                raise DataAccessError("The saved metadata contains a non-object record.")
            saved_count += 1
            mid = doc.get("material_id")
            if not isinstance(mid, str) or mid not in wanted:
                issues.append({"kind": "rejected_saved_metadata_id", "material_id": mid})
                continue
            if mid in materials and fingerprint(materials[mid]) != fingerprint(doc):
                conflicts.add(mid)
            else:
                materials[mid] = doc
        for mid in sorted(conflicts):
            materials.pop(mid, None)
            issues.append({"kind": "conflicting_saved_metadata", "material_id": mid})
    if saved_count < manifest.get("material_metadata_records", 0):
        raise DataAccessError("The metadata file contains fewer saved records than the manifest records.")
    return manifest, documents, list(materials.values()), issues


def recover_snapshot(client: MPClient, root: Path, source: Path) -> Path:
    source = source.resolve()
    original, documents, materials, source_issues = load_recovery_source(source)
    ids = requested_material_ids(documents)
    found = {doc["material_id"] for doc in materials}
    remaining = [mid for mid in ids if mid not in found]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = root / "data" / "raw" / f"mp_{stamp}_{uuid4().hex[:6]}_recovered"
    snapshot.mkdir(parents=True, exist_ok=False)
    manifest = deepcopy(original)
    for key in ("sha256", "failure_type", "finished_at_utc", "stopped_at_utc", "missing_material_metadata_ids"):
        manifest.pop(key, None)
    known_versions = set(original.get("api_reported_database_versions", []))
    if len(known_versions) > 1:
        raise DataAccessError("The saved snapshot reports multiple database versions.")
    client.db_versions.update(known_versions)
    manifest.update({
        "status": "recovering_metadata", "recovery_started_at_utc": now(),
        "recovery_source_snapshot": source.name,
        "recovery_source_sha256": {name: digest(source / name) for name in (
            "manifest.json", "electrodes.jsonl", "materials.jsonl"
        ) if (source / name).exists()},
        "electrode_download_complete": True,
        "material_metadata_records": len(materials),
        "metadata_issues_count": len(source_issues),
        "reused_material_metadata_records": len(materials),
        "metadata_ids_to_retry": len(remaining),
        "database_version_continuity_verified": False,
    })
    manifest.setdefault("notes", []).append(
        "Recovery reuses the original electrode records and exact-ID metadata; unresolved symmetry remains missing."
    )
    manifest["notes"].append(
        "Original v1 failures did not persist the database version. Cross-session version continuity cannot be certified; recovery retrieval versions do not date the original electrode data."
    )
    manifest_path = snapshot / "manifest.json"
    save(manifest_path, manifest)
    try:
        shutil.copyfile(source / "electrodes.jsonl", snapshot / "electrodes.jsonl")
        with (snapshot / "materials.jsonl").open("x", encoding="utf-8") as handle, (
            snapshot / "metadata_lookup_issues.jsonl"
        ).open("x", encoding="utf-8") as issue_handle:
            for doc in materials:
                handle.write(json.dumps(doc, allow_nan=False) + "\n")
            for issue in source_issues:
                issue_handle.write(json.dumps(issue, allow_nan=False) + "\n")
            handle.flush()
            issue_handle.flush()
            print(f"[OK] Reusing {len(documents)} electrode records; no electrode download is needed.", flush=True)
            print(f"[OK] Reusing {len(materials)} exact-ID metadata records.", flush=True)
            print(f"Retrying metadata for {len(remaining)} unresolved endpoint IDs...", flush=True)
            for start in range(0, len(remaining), 100):
                docs, issues = fetch_material_batch(client, remaining[start:start + 100])
                for doc in docs:
                    handle.write(json.dumps(doc, allow_nan=False) + "\n")
                for issue in issues:
                    issue_handle.write(json.dumps(issue, allow_nan=False) + "\n")
                handle.flush()
                issue_handle.flush()
                materials.extend(docs)
                manifest["material_metadata_records"] = len(materials)
                manifest["metadata_issues_count"] += len(issues)
                manifest["api_reported_database_versions"] = sorted(client.db_versions)
                save(manifest_path, manifest)
                print(f"  Checked {min(start + 100, len(remaining))}/{len(remaining)} retry IDs; {len(materials)} verified metadata records total", flush=True)
        found = {doc["material_id"] for doc in materials}
        manifest["missing_material_metadata_ids"] = sorted(set(ids) - found)
        manifest["sha256"] = {name: digest(snapshot / name) for name in (
            "electrodes.jsonl", "materials.jsonl", "metadata_lookup_issues.jsonl"
        )}
        manifest["finished_at_utc"] = now()
        manifest["status"] = "complete"
        save(manifest_path, manifest)
        print(f"Unresolved symmetry metadata IDs: {len(manifest['missing_material_metadata_ids'])} (flagged in the report).", flush=True)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["failure_type"] = type(exc).__name__
        manifest["stopped_at_utc"] = now()
        save(manifest_path, manifest)
        print(f"Incomplete recovery preserved at: {snapshot}", flush=True)
        raise
    print(f"Recovered snapshot: {snapshot}", flush=True)
    return snapshot
