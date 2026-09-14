"""Offline recovery regressions. All records here are synthetic, not research data."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from src.data.inspection import inspect_snapshot
from src.data.mp_download import (
    BASE_URL, ELECTRODE_ROUTE, MATERIAL_ROUTE, DataAccessError, MPClient,
    fetch_material_batch, requested_material_ids,
)
from src.data.recovery import load_recovery_source, recover_snapshot
from test_step3_data import FakeOpener, electrode, http_error, materials, payload


class MetadataResponseTests(unittest.TestCase):
    def client(self, responses):
        self.opener = FakeOpener(responses)
        return MPClient("fake-secret-for-tests", opener=self.opener, progress=lambda _: None)

    def test_rejected_id_is_not_assigned_to_requested_id(self):
        client = self.client([payload([{"material_id": "mp-unrelated"}])])
        docs, issues = fetch_material_batch(client, ["mp-requested"])
        self.assertEqual(docs, [])
        self.assertEqual(issues[-1], {"kind": "unresolved_material_metadata", "material_id": "mp-requested"})

    def test_mismatch_batch_retries_missing_id_singly_and_recovers(self):
        a, b = materials()
        client = self.client([payload([a, {"material_id": "mp-unrelated"}]), payload([b])])
        docs, issues = fetch_material_batch(client, [a["material_id"], b["material_id"]])
        self.assertEqual(docs, [a, b])
        self.assertEqual(issues[0]["rejected_material_ids"], ["mp-unrelated"])
        query = parse_qs(urlsplit(self.opener.requests[1].full_url).query)
        self.assertEqual(query["material_ids"], [b["material_id"]])

    def test_ignored_filter_does_not_page_through_whole_database(self):
        a, b = materials()
        client = self.client([
            payload([{"material_id": "mp-unrelated"}], total=100000), payload([a]), payload([b]),
        ])
        docs, _ = fetch_material_batch(client, [a["material_id"], b["material_id"]])
        self.assertEqual(docs, [a, b])
        self.assertEqual(len(self.opener.requests), 3)
        self.assertTrue(all(parse_qs(urlsplit(req.full_url).query)["_skip"] == ["0"] for req in self.opener.requests))

    def test_empty_summary_is_missing_without_fabricating_symmetry(self):
        docs, issues = fetch_material_batch(self.client([payload([])]), ["mp-absent"])
        self.assertEqual(docs, [])
        self.assertEqual(issues[0]["kind"], "unresolved_material_metadata")

    def test_duplicate_conflicting_metadata_is_not_accepted(self):
        a = materials()[0]
        conflict = dict(a, symmetry={"number": 1, "crystal_system": "Triclinic"})
        docs, issues = fetch_material_batch(self.client([payload([a, conflict])]), [a["material_id"]])
        self.assertEqual(docs, [])
        self.assertEqual(issues[0]["conflicting_material_ids"], [a["material_id"]])

    def test_authentication_error_is_not_reported_as_missing_metadata(self):
        with self.assertRaisesRegex(DataAccessError, "403"):
            fetch_material_batch(self.client([http_error(403)]), ["mp-requested"])

    def test_invalid_total_is_not_silently_accepted(self):
        with self.assertRaisesRegex(DataAccessError, "Invalid metadata response total"):
            fetch_material_batch(self.client([{"data": [], "meta": {}}]), ["mp-requested"])


class RecoveryTests(unittest.TestCase):
    def source(self, root):
        source = root / "data" / "raw" / "failed_original"
        source.mkdir(parents=True)
        doc = electrode()
        first = materials()[0]
        manifest = {
            "schema_version": "step3-v1", "status": "failed", "failure_type": "DataAccessError",
            "source": BASE_URL, "electrode_route": ELECTRODE_ROUTE, "material_route": MATERIAL_ROUTE,
            "started_at_utc": "2026-09-13T17:03:43Z", "requested_ions": ["Li"],
            "electrode_records": 1, "electrode_records_by_ion": {"Li": 1},
            "requested_material_ids": len(requested_material_ids([doc])),
            "material_metadata_records": 1,
            "page_log": [{"route": "electrodes", "working_ion": "Li", "skip": 0, "count": 1, "total": 1}],
            "notes": [],
        }
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (source / "electrodes.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
        (source / "materials.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
        return source

    def client(self, responses):
        self.opener = FakeOpener(responses)
        return MPClient("fake-secret-for-tests", opener=self.opener, progress=lambda _: None)

    def test_recovery_preserves_source_and_only_queries_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new_callable=io.StringIO):
            root = Path(directory)
            source = self.source(root)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            snapshot = recover_snapshot(self.client([payload([materials()[1]])]), root, source)
            report, _ = inspect_snapshot(root, snapshot)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertNotEqual(source, snapshot)
            self.assertEqual(len(self.opener.requests), 1)
            self.assertIn(MATERIAL_ROUTE, self.opener.requests[0].full_url)
            self.assertNotIn(ELECTRODE_ROUTE, self.opener.requests[0].full_url)
            self.assertEqual(report["adjacent_intervals"]["complete_basic_inputs"], 1)
            self.assertEqual(report["recovery"]["reused_material_metadata_records"], 1)
            self.assertFalse(report["recovery"]["database_version_continuity_verified"])
            for path in snapshot.iterdir():
                self.assertNotIn("fake-secret-for-tests", path.read_text(encoding="utf-8"))

    def test_unresolved_id_keeps_actual_voltage_and_flags_missing_symmetry(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new_callable=io.StringIO):
            root = Path(directory)
            source = self.source(root)
            snapshot = recover_snapshot(self.client([payload([{"material_id": "mp-unrelated"}])]), root, source)
            report, _ = inspect_snapshot(root, snapshot)
            self.assertEqual(report["missing_material_metadata_ids"], [materials()[1]["material_id"]])
            self.assertEqual(report["adjacent_intervals"]["complete_basic_inputs"], 0)
            csv_path = root / "data" / "processed" / snapshot.name / "voltage_intervals.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["average_voltage_V"], "3.4")
            self.assertEqual(row["spacegroup_number_discharge"], "")
            self.assertIn("missing_material_metadata_discharge", row["warnings"])

    def test_partial_last_metadata_line_is_discarded_and_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            with (source / "materials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"material_id":')
            _, _, docs, issues = load_recovery_source(source)
            self.assertEqual(len(docs), 1)
            self.assertEqual(issues[0]["kind"], "discarded_unfinished_metadata_line")

    def test_valid_metadata_written_after_last_manifest_checkpoint_is_reused(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new_callable=io.StringIO):
            root = Path(directory)
            source = self.source(root)
            with (source / "materials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(materials()[1]) + "\n")
            snapshot = recover_snapshot(self.client([]), root, source)
            report, _ = inspect_snapshot(root, snapshot)
            self.assertEqual(len(self.opener.requests), 0)
            self.assertEqual(report["recovery"]["reused_material_metadata_records"], 2)

    def test_incomplete_electrode_download_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            path = source / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["page_log"][0]["total"] = 2
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DataAccessError, "page sequence"):
                recover_snapshot(self.client([]), Path(directory), source)
            self.assertEqual(self.opener.requests, [])

    def test_wrong_electrode_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            (source / "electrodes.jsonl").write_text("")
            with self.assertRaisesRegex(DataAccessError, "counts"):
                load_recovery_source(source)

    def test_damaged_middle_metadata_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory))
            with (source / "materials.jsonl").open("a") as handle:
                handle.write('not-json\n' + json.dumps(materials()[1]) + '\n')
            with self.assertRaisesRegex(DataAccessError, "damaged line"):
                load_recovery_source(source)

    def test_network_failure_remains_failed_and_can_be_recovered_again(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new_callable=io.StringIO):
            root = Path(directory)
            source = self.source(root)
            with self.assertRaises(DataAccessError):
                recover_snapshot(self.client([http_error(403)]), root, source)
            failed = next(p for p in (root / "data" / "raw").iterdir() if p != source)
            self.assertEqual(json.loads((failed / "manifest.json").read_text())["status"], "failed")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                inspect_snapshot(root, failed)
            recovered = recover_snapshot(self.client([payload([materials()[1]])]), root, failed)
            report, _ = inspect_snapshot(root, recovered)
            self.assertEqual(report["adjacent_intervals"]["complete_basic_inputs"], 1)

    def test_recorded_database_version_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new_callable=io.StringIO):
            root = Path(directory)
            source = self.source(root)
            path = source / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["api_reported_database_versions"] = ["old-version"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DataAccessError, "version changed"):
                recover_snapshot(self.client([payload([materials()[1]], version="new-version")]), root, source)


if __name__ == "__main__":
    unittest.main()
