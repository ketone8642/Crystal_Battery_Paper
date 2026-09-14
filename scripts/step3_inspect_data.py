"""Rebuild inspection tables from an existing complete snapshot, without an API key."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.inspection import inspect_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest complete snapshot")
    args = parser.parse_args()
    try:
        snapshot = args.snapshot
        if snapshot is None:
            candidates = []
            for path in (ROOT / "data" / "raw").glob("mp_*/manifest.json"):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") == "complete":
                    candidates.append(path.parent)
            if not candidates:
                raise ValueError("No complete snapshot found. Run step3_collect_data.py first.")
            snapshot = sorted(candidates)[-1]
        elif not snapshot.is_absolute():
            snapshot = ROOT / snapshot
        inspect_snapshot(ROOT, snapshot)
        return 0
    except (ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
