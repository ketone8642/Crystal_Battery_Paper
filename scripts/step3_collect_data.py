"""Download a new Materials Project snapshot and inspect its actual labels.

No additional packages are required. Run with --check for a small API access
check, or without --check to download the requested ions. Use --recover PATH
to reuse a failed snapshot whose electrode download finished. Keys are never saved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from uuid import uuid4
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.inspection import inspect_snapshot
from src.data.mp_download import (
    BASE_URL, ELECTRODE_FIELDS, ELECTRODE_ROUTE, IONS, MATERIAL_FIELDS,
    MATERIAL_ROUTE, DataAccessError, MPClient, element_symbol, fetch_material_batch,
    requested_material_ids,
)
from src.data.recovery import recover_snapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_key() -> str:
    key = os.environ.get("MP_API_KEY", "").strip()
    if key:
        return key
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("Materials Project API key (input hidden): ").strip()
    except getpass.GetPassWarning:
        raise DataAccessError("Hidden input is unavailable in this console. Run the command in a normal PowerShell terminal.") from None
    except EOFError:
        raise DataAccessError("The API-key prompt was closed. Run the command again in PowerShell.") from None
    if not key:
        raise DataAccessError("No key was entered. Obtain your key from your Materials Project dashboard and run again.")
    return key


def check_connection(client: MPClient, ion: str) -> None:
    params = {"working_ion": ion, "_fields": ",".join(ELECTRODE_FIELDS), "_limit": 1}
    payload = client.get(ELECTRODE_ROUTE, params)
    total = (payload.get("meta") or {}).get("total_doc")
    print("[OK] Materials Project API request succeeded.")
    print(f"API-reported {ion} electrode count: {total}")
    if not payload["data"]:
        print("No records returned for this ion. Choose another supported ion with --ions Li.")
        return
    first = payload["data"][0]
    if element_symbol(first.get("working_ion")) != ion:
        raise DataAccessError("The API returned an unexpected working ion. The filtering/schema needs review.")
    print("Returned fields: " + ", ".join(sorted(first)))
    pairs = first.get("adj_pairs")
    print(f"Adjacent intervals in this first record: {len(pairs) if isinstance(pairs, list) else 'not provided'}")
    print("API CHECK COMPLETE. No dataset has been downloaded or model trained in this check.")


def collect(client: MPClient, ions: list[str], page_size: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = ROOT / "data" / "raw" / f"mp_{stamp}_{uuid4().hex[:6]}"
    snapshot.mkdir(parents=True, exist_ok=False)
    manifest_path = snapshot / "manifest.json"
    manifest = {
        "schema_version": "step3-v1",
        "status": "downloading",
        "started_at_utc": utc_now(),
        "source": BASE_URL,
        "electrode_route": ELECTRODE_ROUTE,
        "material_route": MATERIAL_ROUTE,
        "requested_ions": ions,
        "electrode_fields": list(ELECTRODE_FIELDS),
        "material_fields": list(MATERIAL_FIELDS),
        "page_size": page_size,
        "python_version": platform.python_version(),
        "electrode_records": 0,
        "material_metadata_records": 0,
        "electrode_records_by_ion": {},
        "page_log": [],
        "notes": [
            "Projected raw API records are preserved in JSONL. No API key is saved.",
            "This is a current database snapshot, not the paper's original 2019 data.",
            "Na and K are additional ions, absent from the paper's main training set.",
            "A download snapshot is not a training/test split.",
        ],
    }
    save_manifest(manifest_path, manifest)
    documents: list[dict] = []
    materials: list[dict] = []
    try:
        with (snapshot / "electrodes.jsonl").open("x", encoding="utf-8") as handle:
            for ion in ions:
                ion_count = 0
                print(f"\nDownloading {ion} electrode records...", flush=True)
                params = {"working_ion": ion, "_fields": ",".join(ELECTRODE_FIELDS), "_sort_fields": "battery_id"}
                for docs, page in client.pages(ELECTRODE_ROUTE, params, page_size=page_size):
                    for doc in docs:
                        if element_symbol(doc.get("working_ion")) != ion:
                            raise DataAccessError("The API returned a working ion that does not match the requested filter.")
                        handle.write(json.dumps(doc, allow_nan=False) + "\n")
                    handle.flush()
                    documents.extend(docs)
                    ion_count += len(docs)
                    manifest["electrode_records"] = len(documents)
                    manifest["electrode_records_by_ion"][ion] = ion_count
                    manifest["page_log"].append({"route": "electrodes", "working_ion": ion, **page})
                    manifest["api_reported_database_versions"] = sorted(client.db_versions)
                    save_manifest(manifest_path, manifest)
                    print(f"  {ion}: {ion_count}/{page['total']} records saved", flush=True)
                manifest["electrode_records_by_ion"][ion] = ion_count
                if ion_count == 0:
                    print(f"  No {ion} records returned. This absence will be reported.", flush=True)
        if not documents:
            raise DataAccessError("No electrode records were returned for the requested ions.")

        ids = requested_material_ids(documents)
        manifest["requested_material_ids"] = len(ids)
        manifest["electrode_download_complete"] = True
        manifest["metadata_issues_count"] = 0
        save_manifest(manifest_path, manifest)
        print(f"\nRetrieving symmetry metadata for {len(ids)} endpoint material IDs...", flush=True)
        with (snapshot / "materials.jsonl").open("x", encoding="utf-8") as handle, (
            snapshot / "metadata_lookup_issues.jsonl"
        ).open("x", encoding="utf-8") as issue_handle:
            for start in range(0, len(ids), 100):
                batch = ids[start:start + 100]
                docs, issues = fetch_material_batch(client, batch)
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
                manifest["page_log"].append({"route": "materials", "batch_start": start, "requested_count": len(batch), "count": len(docs)})
                save_manifest(manifest_path, manifest)
                print(f"  Queried {min(start + 100, len(ids))}/{len(ids)} IDs; received {len(materials)} metadata records", flush=True)
        found = {doc.get("material_id") for doc in materials}
        manifest["missing_material_metadata_ids"] = sorted(set(ids) - found)
        manifest["api_reported_database_versions"] = sorted(client.db_versions)
        manifest["sha256"] = {
            name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            for name in ("electrodes.jsonl", "materials.jsonl", "metadata_lookup_issues.jsonl")
        }
        manifest["finished_at_utc"] = utc_now()
        manifest["status"] = "complete"
        save_manifest(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["stopped_at_utc"] = utc_now()
        # Do not put exception strings, request headers or credentials in reports.
        manifest["failure_type"] = type(exc).__name__
        manifest["api_reported_database_versions"] = sorted(client.db_versions)
        save_manifest(manifest_path, manifest)
        print(f"Incomplete snapshot preserved at: {snapshot}", flush=True)
        raise
    print(f"\nRaw snapshot saved: {snapshot}", flush=True)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check API access with one electrode record")
    mode.add_argument("--recover", type=Path, help="Recover metadata from a failed snapshot with complete electrode records")
    parser.add_argument("--ions", nargs="+", choices=IONS, default=list(IONS))
    parser.add_argument("--page-size", type=int, default=200, help="Electrode records per request (1-1000)")
    args = parser.parse_args()
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size must be between 1 and 1000")
    ions = list(dict.fromkeys(args.ions))
    try:
        client = MPClient(read_key(), progress=lambda message: print(message, flush=True))
        if args.check:
            check_connection(client, ions[0])
            return 0
        snapshot = (recover_snapshot(client, ROOT, args.recover) if args.recover
                    else collect(client, ions, args.page_size))
        report, _ = inspect_snapshot(ROOT, snapshot)
        print("\nSTEP 3 DOWNLOAD AND INSPECTION COMPLETE")
        if report["adjacent_intervals"]["complete_basic_inputs"] == 0:
            print("ATTENTION: No adjacent interval has all basic inputs. Share the dataset report before continuing.")
        print("Share dataset_report.json for review before Step 4.")
        return 0
    except KeyboardInterrupt:
        print("\nDownload interrupted. No completed dataset is claimed. Run the command again when ready.")
        return 130
    except (DataAccessError, ValueError, OSError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
