"""Offline software tests. These invented records are NOT research data."""

from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from src.data.inspection import inspect_documents, inspect_snapshot, make_row
from src.data.mp_download import (
    DataAccessError, ELECTRODE_ROUTE, MATERIAL_ROUTE, MPClient,
    requested_material_ids,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("step3_collect", ROOT / "scripts" / "step3_collect_data.py")
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def electrode():
    pair = {
        "formula_charge": "FePO4", "formula_discharge": "LiFePO4",
        "id_charge": "mp-test-charge", "id_discharge": "mp-test-discharge",
        "fracA_charge": 0.0, "fracA_discharge": 1 / 7,
        "average_voltage": 3.4,
    }
    return {
        "battery_id": "synthetic-test-only", "working_ion": "Li",
        "framework_formula": "FePO4", "num_steps": 1,
        "last_updated": "2026-01-01T00:00:00Z", "warnings": [],
        **pair, "adj_pairs": [dict(pair)],
    }


def materials():
    return [
        {"material_id": name, "symmetry": {"crystal_system": "Orthorhombic", "number": 62}, "deprecated": False}
        for name in ("mp-test-charge", "mp-test-discharge")
    ]


def payload(docs, total=None, version="test-version"):
    return {"data": docs, "meta": {"total_doc": len(docs) if total is None else total, "db_version": version}}


class FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, req, timeout):
        self.requests.append(req)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode())


def http_error(code, detail="failure", headers=None):
    return HTTPError("https://api.materialsproject.org/", code, "test", headers or {}, io.BytesIO(detail.encode()))


class ClientTests(unittest.TestCase):
    def client(self, responses):
        self.opener = FakeOpener(responses)
        self.delays = []
        return MPClient("fake-secret-for-tests", opener=self.opener, sleep=self.delays.append, progress=lambda _: None)

    def test_authentication_and_actual_page_offsets(self):
        client = self.client([payload([{"id": 1}], 3), payload([{"id": 2}, {"id": 3}], 3)])
        pages = list(client.pages(ELECTRODE_ROUTE, {"working_ion": "Li"}, page_size=2))
        self.assertEqual([page[1]["skip"] for page in pages], [0, 1])
        self.assertEqual(len(self.opener.requests), 2)
        req = self.opener.requests[1]
        self.assertEqual(req.get_header("X-api-key"), "fake-secret-for-tests")
        self.assertNotIn("fake-secret", req.full_url)
        self.assertEqual(parse_qs(urlsplit(req.full_url).query)["_skip"], ["1"])

    def test_access_denied_does_not_retry_or_leak_key(self):
        for status in (401, 403):
            with self.subTest(status=status):
                client = self.client([http_error(status, "fake-secret-for-tests")])
                with self.assertRaises(DataAccessError) as caught:
                    client.get(ELECTRODE_ROUTE, {})
                self.assertIn(str(status), str(caught.exception))
                self.assertNotIn("fake-secret-for-tests", str(caught.exception))
                self.assertEqual(self.delays, [])

    def test_schema_error_redacts_key(self):
        client = self.client([http_error(422, "invalid fake-secret-for-tests")])
        with self.assertRaises(DataAccessError) as caught:
            client.get(ELECTRODE_ROUTE, {})
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertNotIn("fake-secret-for-tests", str(caught.exception))

    def test_rate_limit_retries_and_caps_wait(self):
        client = self.client([http_error(429, headers={"Retry-After": "999"}), payload([])])
        self.assertEqual(client.get(ELECTRODE_ROUTE, {})["data"], [])
        self.assertEqual(self.delays, [30])

    def test_premature_empty_page_is_rejected(self):
        client = self.client([payload([{"id": 1}], 2), payload([], 2)])
        with self.assertRaisesRegex(DataAccessError, "empty page"):
            list(client.pages(ELECTRODE_ROUTE, {}, page_size=1))

    def test_repeated_page_is_rejected(self):
        client = self.client([payload([{"id": 1}], 2), payload([{"id": 1}], 2)])
        with self.assertRaisesRegex(DataAccessError, "repeated a page"):
            list(client.pages(ELECTRODE_ROUTE, {}, page_size=1))

    def test_changed_total_is_rejected(self):
        client = self.client([payload([{"id": 1}], 2), payload([{"id": 2}], 3)])
        with self.assertRaisesRegex(DataAccessError, "count changed"):
            list(client.pages(ELECTRODE_ROUTE, {}, page_size=1))

    def test_changed_database_version_is_rejected(self):
        client = self.client([payload([], version="v1"), payload([], version="v2")])
        client.get(ELECTRODE_ROUTE, {})
        with self.assertRaisesRegex(DataAccessError, "version changed"):
            client.get(MATERIAL_ROUTE, {})

    def test_missing_total_cannot_claim_complete_download(self):
        client = self.client([{"data": []}])
        with self.assertRaisesRegex(DataAccessError, "total_doc"):
            list(client.pages(ELECTRODE_ROUTE, {}))

    def test_zero_record_query_is_complete(self):
        client = self.client([payload([])])
        self.assertEqual(list(client.pages(ELECTRODE_ROUTE, {})), [])

    def test_unknown_route_is_rejected_before_network(self):
        client = self.client([])
        with self.assertRaisesRegex(DataAccessError, "Unsupported"):
            client.get("https://example.com/", {})
        self.assertEqual(self.opener.requests, [])


class InspectionTests(unittest.TestCase):
    def test_overall_and_adjacent_tables_are_separate(self):
        summaries, intervals, report = inspect_documents([electrode()], materials())
        self.assertEqual(len(summaries), 1)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(summaries[0]["row_type"], "overall")
        self.assertEqual(intervals[0]["row_type"], "adjacent")
        self.assertEqual(report["adjacent_intervals"]["complete_basic_inputs"], 1)

    def test_missing_interval_values_are_never_copied_from_parent(self):
        doc = electrode()
        doc["adj_pairs"] = [{}]
        _, intervals, _ = inspect_documents([doc], materials())
        for name in ("average_voltage_V", "formula_charge", "id_charge", "fracA_charge"):
            self.assertIsNone(intervals[0][name])
        self.assertFalse(intervals[0]["basic_checks_passed"])

    def test_zero_and_negative_voltage_are_retained(self):
        for voltage in (0.0, -0.8):
            with self.subTest(voltage=voltage):
                doc = electrode()
                doc["adj_pairs"][0]["average_voltage"] = voltage
                _, rows, _ = inspect_documents([doc], materials())
                self.assertTrue(rows[0]["basic_checks_passed"])
                self.assertEqual(rows[0]["average_voltage_V"], voltage)

    def test_invalid_targets_are_flagged(self):
        for voltage in (None, True, "3.4", float("nan"), float("inf")):
            with self.subTest(voltage=voltage):
                doc = electrode()
                pair = dict(doc["adj_pairs"][0], average_voltage=voltage)
                row = make_row(doc, pair, "adjacent", 0, {})
                self.assertFalse(row["basic_checks_passed"])
                self.assertIsNone(row["average_voltage_V"])

    def test_invalid_fractions_are_flagged(self):
        for low, high in ((0.2, 0.1), (0, 0), (-0.1, 0.2), (0, 1), (False, 0.2)):
            with self.subTest(low=low, high=high):
                doc = electrode()
                doc["adj_pairs"][0].update(fracA_charge=low, fracA_discharge=high)
                _, rows, _ = inspect_documents([doc], materials())
                self.assertFalse(rows[0]["basic_checks_passed"])

    def test_missing_metadata_is_reported_without_inventing_symmetry(self):
        _, rows, _ = inspect_documents([electrode()], [])
        self.assertTrue(rows[0]["basic_checks_passed"])
        self.assertFalse(rows[0]["complete_basic_inputs"])
        self.assertIsNone(rows[0]["spacegroup_number_charge"])
        self.assertIn("missing_material_metadata_charge", rows[0]["warnings"])

    def test_exact_duplicate_source_documents_are_counted_and_collapsed(self):
        doc = electrode()
        summaries, intervals, report = inspect_documents([doc, deepcopy(doc)], materials())
        self.assertEqual((len(summaries), len(intervals)), (1, 1))
        self.assertEqual(report["exact_duplicate_documents_collapsed"], 1)

    def test_conflicting_interval_labels_are_flagged_on_both_rows(self):
        first = electrode()
        second = deepcopy(first)
        second["battery_id"] = "different-source"
        second["adj_pairs"][0]["average_voltage"] = 3.8
        _, rows, report = inspect_documents([first, second], materials())
        self.assertEqual(report["conflicting_interval_label_groups"], 1)
        self.assertTrue(all(not row["basic_checks_passed"] for row in rows))
        self.assertTrue(all("conflicting_interval_labels" in row["issues"] for row in rows))

    def test_missing_adjacent_pairs_do_not_create_synthetic_intervals(self):
        doc = electrode()
        doc.pop("adj_pairs")
        summaries, intervals, report = inspect_documents([doc], materials())
        self.assertEqual(len(summaries), 1)
        self.assertEqual(intervals, [])
        self.assertEqual(report["documents_without_adjacent_pairs"], 1)

    def test_material_ids_include_intermediate_endpoints(self):
        doc = electrode()
        doc["adj_pairs"].append({"id_charge": "mp-test-middle", "id_discharge": "mp-test-discharge"})
        self.assertEqual(requested_material_ids([doc]), ["mp-test-charge", "mp-test-discharge", "mp-test-middle"])


class SnapshotTests(unittest.TestCase):
    def make_snapshot(self, root):
        client = MPClient("fake-secret-for-tests", opener=FakeOpener([payload([electrode()]), payload(materials())]))
        with patch.object(COLLECTOR, "ROOT", root), patch("sys.stdout", new_callable=io.StringIO):
            return COLLECTOR.collect(client, ["Li"], 200)

    def test_complete_offline_pipeline_writes_actual_counts_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.make_snapshot(root)
            with patch("sys.stdout", new_callable=io.StringIO):
                report, path = inspect_snapshot(root, snapshot)
            self.assertTrue(path.is_file())
            self.assertEqual(report["raw_electrode_records"], 1)
            self.assertEqual(report["adjacent_intervals"]["rows"], 1)
            csv_path = root / "data" / "processed" / snapshot.name / "voltage_intervals.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["average_voltage_V"], "3.4")
            self.assertEqual(rows[0]["complete_basic_inputs"], "True")
            for file in root.rglob("*"):
                if file.is_file():
                    self.assertNotIn("fake-secret-for-tests", file.read_text(encoding="utf-8-sig"))

    def test_modified_snapshot_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.make_snapshot(root)
            raw = snapshot / "electrodes.jsonl"
            raw.write_text(raw.read_text(encoding="utf-8").replace("3.4", "9.9"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                inspect_snapshot(root, snapshot)

    def test_incomplete_snapshot_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "data" / "raw" / "incomplete"
            snapshot.mkdir(parents=True)
            (snapshot / "manifest.json").write_text('{"status":"failed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                inspect_snapshot(root, snapshot)

    def test_mid_download_failure_is_recorded_without_complete_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = MPClient("fake-secret-for-tests", opener=FakeOpener([payload([electrode()]), http_error(403)]))
            with patch.object(COLLECTOR, "ROOT", root), patch("sys.stdout", new_callable=io.StringIO):
                with self.assertRaises(DataAccessError):
                    COLLECTOR.collect(client, ["Li"], 200)
            manifests = list((root / "data" / "raw").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["electrode_records"], 1)
            self.assertNotIn("fake-secret-for-tests", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
