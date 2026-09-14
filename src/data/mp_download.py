"""Small read-only client for the official Materials Project REST API.

Uses Python's standard library. No mp-api/pymatgen installation is required for
this data-inspection stage. The API route and fields follow the official client
and Emmet electrode model; see README_STEP3.md for source links.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable, Iterator
from urllib import error, parse, request


BASE_URL = "https://api.materialsproject.org/"
ELECTRODE_ROUTE = "materials/insertion_electrodes/"
MATERIAL_ROUTE = "materials/summary/"
IONS = ("Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y")
ELECTRODE_FIELDS = (
    "battery_id", "working_ion", "framework_formula", "material_ids",
    "formula_charge", "formula_discharge", "id_charge", "id_discharge",
    "fracA_charge", "fracA_discharge", "average_voltage", "num_steps",
    "adj_pairs", "last_updated", "warnings",
)
MATERIAL_FIELDS = ("material_id", "formula_pretty", "symmetry", "deprecated")


class DataAccessError(RuntimeError):
    """An actionable retrieval failure with no authentication key in its text."""


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Prevent an authentication header from being forwarded to another host.
        return None


def fingerprint(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def element_symbol(value) -> str | None:
    if isinstance(value, str) and value in IONS:
        return value
    if isinstance(value, dict):
        for name in ("element", "symbol", "value"):
            if value.get(name) in IONS:
                return value[name]
    return None


class MPClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 60,
        attempts: int = 3,
        opener=None,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[str], None] = print,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise DataAccessError("An API key is required. Run the command again and paste it at the hidden prompt.")
        self._api_key = api_key.strip()
        self.timeout = timeout
        self.attempts = attempts
        self._opener = opener or request.build_opener(NoRedirect())
        self._sleep = sleep
        self._progress = progress
        self.db_versions: set[str] = set()

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]")

    def get(self, route: str, params: dict) -> dict:
        if route not in (ELECTRODE_ROUTE, MATERIAL_ROUTE):
            raise DataAccessError("Unsupported Materials Project route.")
        url = BASE_URL + route + "?" + parse.urlencode(params)
        for attempt in range(self.attempts):
            req = request.Request(url, headers={
                "X-API-KEY": self._api_key,
                "Accept": "application/json",
                "User-Agent": "BatteryVoltageTeachingProject/0.2",
            })
            try:
                with self._opener.open(req, timeout=self.timeout) as response:
                    # A corrupt/oversized response must not exhaust memory.
                    body = response.read(64 * 1024 * 1024 + 1)
                    if len(body) > 64 * 1024 * 1024:
                        raise DataAccessError("One response exceeds 64 MiB. Retry with --page-size 50.")
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    raise DataAccessError("The API returned an unexpected response schema: a data list is required.")
                if any(not isinstance(doc, dict) for doc in payload["data"]):
                    raise DataAccessError("The API returned a non-object electrode/material record.")
                metadata = payload.get("meta") or {}
                if not isinstance(metadata, dict):
                    raise DataAccessError("The API returned invalid metadata.")
                db_version = metadata.get("db_version")
                if db_version is not None:
                    self.db_versions.add(str(db_version))
                    if len(self.db_versions) > 1:
                        raise DataAccessError("The API database version changed during the download. Retry to create one consistent snapshot.")
                return payload
            except error.HTTPError as exc:
                status = exc.code
                if status in (401, 403):
                    exc.close()
                    raise DataAccessError(f"HTTP {status}: access denied. Verify your Materials Project API key and account access.") from None
                if status in (429, 500, 502, 503, 504) and attempt + 1 < self.attempts:
                    retry_after = exc.headers.get("Retry-After", "")
                    exc.close()
                    try:
                        delay = float(retry_after)
                        if not math.isfinite(delay):
                            raise ValueError
                    except (TypeError, ValueError):
                        delay = 2 ** (attempt + 1)
                    delay = max(1, min(delay, 30))
                    self._progress(f"HTTP {status}; retrying in {delay:g} seconds...")
                    self._sleep(delay)
                    continue
                detail = self._redact(exc.read(1500).decode("utf-8", errors="replace"))
                exc.close()
                if status == 422:
                    message = "The API rejected the query or requested fields. Share this error so the schema can be checked."
                elif status in (301, 302, 303, 307, 308, 404, 410):
                    message = "The configured official API route may have changed. Share this error."
                else:
                    message = "The API request failed after the available attempts."
                raise DataAccessError(f"HTTP {status}: {message}\n{detail}") from None
            except (error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt + 1 < self.attempts:
                    self._progress(f"Connection problem; retrying request ({attempt + 2}/{self.attempts})...")
                    self._sleep(2 ** (attempt + 1))
                    continue
                reason = self._redact(str(getattr(exc, "reason", exc)))
                raise DataAccessError(f"Cannot reach Materials Project: {reason}. Check your connection/proxy and retry.") from None
            except (ValueError, UnicodeError):
                raise DataAccessError("The API response was not valid JSON. Check access to api.materialsproject.org.") from None
        raise DataAccessError("The API request did not complete.")

    def pages(self, route: str, params: dict, *, page_size: int = 200) -> Iterator[tuple[list[dict], dict]]:
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        offset = 0
        expected_total = None
        seen_pages: set[str] = set()
        while True:
            query = {**params, "_limit": page_size, "_skip": offset}
            payload = self.get(route, query)
            total = (payload.get("meta") or {}).get("total_doc")
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                raise DataAccessError("The API omitted a valid meta.total_doc; download completeness cannot be verified.")
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise DataAccessError("The record count changed during pagination. Retry to avoid an incomplete snapshot.")
            docs = payload["data"]
            if not docs:
                if offset != total:
                    raise DataAccessError("The API returned an empty page before all records were downloaded.")
                return
            signature = fingerprint({"records": docs})
            if signature in seen_pages:
                raise DataAccessError("The API repeated a page. Stopping to avoid silently duplicating the dataset.")
            seen_pages.add(signature)
            if offset + len(docs) > total:
                raise DataAccessError("The API returned more records than its declared total.")
            yield docs, {"skip": offset, "count": len(docs), "total": total}
            offset += len(docs)
            if offset == total:
                return


def requested_material_ids(documents: list[dict]) -> list[str]:
    ids: set[str] = set()
    for doc in documents:
        pairs = doc.get("adj_pairs")
        candidates = [doc] + (pairs if isinstance(pairs, list) else [])
        for candidate in candidates:
            if isinstance(candidate, dict):
                for field in ("id_charge", "id_discharge"):
                    value = candidate.get(field)
                    if isinstance(value, str) and value:
                        ids.add(value)
    return sorted(ids)


def fetch_material_batch(client: MPClient, batch: list[str]) -> tuple[list[dict], list[dict]]:
    """Keep exact-ID matches only; retry absent IDs singly and record anomalies.

    One response is sufficient for <=100 distinct requested IDs. Asking for one
    extra record detects an ignored filter without paging through the database.
    A different ID is never treated as an alias without independent evidence.
    Network/auth/schema failures still raise rather than being called missing data.
    """
    batch = list(dict.fromkeys(batch))
    if not 1 <= len(batch) <= 100 or any(not isinstance(mid, str) or not mid for mid in batch):
        raise ValueError("A metadata batch requires 1 to 100 nonempty material IDs.")
    issues: list[dict] = []

    def query(ids: list[str]) -> dict[str, dict]:
        result = client.get(MATERIAL_ROUTE, {
            "material_ids": ",".join(ids), "_fields": ",".join(MATERIAL_FIELDS),
            "_sort_fields": "material_id", "_limit": len(ids) + 1, "_skip": 0,
        })
        docs = result["data"]
        total = (result.get("meta") or {}).get("total_doc")
        if isinstance(total, bool) or not isinstance(total, int) or total < len(docs):
            raise DataAccessError("Invalid metadata response total; completeness cannot be assessed.")
        wanted = set(ids)
        matching: dict[str, dict] = {}
        rejected: list = []
        conflicts: set[str] = set()
        for doc in docs:
            mid = doc.get("material_id")
            if not isinstance(mid, str) or mid not in wanted:
                rejected.append(mid)
                continue
            if mid in matching and fingerprint(matching[mid]) != fingerprint(doc):
                conflicts.add(mid)
            else:
                matching[mid] = doc
        for mid in conflicts:
            matching.pop(mid, None)
        if rejected or conflicts or total != len(docs):
            issues.append({
                "kind": "metadata_response_anomaly", "requested_ids": ids,
                "rejected_material_ids": rejected, "conflicting_material_ids": sorted(conflicts),
                "api_total_doc": total, "returned_records": len(docs),
            })
        return matching

    accepted = query(batch)
    missing = [mid for mid in batch if mid not in accepted]
    if missing and len(batch) > 1:
        client._progress(f"  Retrying {len(missing)} unresolved metadata IDs individually... ")
        for mid in missing:
            accepted.update(query([mid]))
    for mid in batch:
        if mid not in accepted:
            issues.append({"kind": "unresolved_material_metadata", "material_id": mid})
    return [accepted[mid] for mid in batch if mid in accepted], issues
