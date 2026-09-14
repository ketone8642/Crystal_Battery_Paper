# Step 3 metadata recovery for Windows PowerShell

This repair addresses the downloader stopping with:

```text
The API returned an unexpected material ID for a filtered metadata request.
```

The check detected at least one returned material ID outside the requested batch.
The original log does not show those IDs or establish the API-side cause. This
repair records future anomalies by requested and returned ID. A different ID is
never assumed to be an alias for an electrode endpoint.

## Install the repair

1. Extract `battery_voltage_step3_recovery_fix.zip` into a temporary folder.
2. Copy its `scripts`, `src`, `tests` folders and README files into your existing
   `battery_voltage_project` folder. Merge the folders and replace the included
   code files when Windows asks.
3. Check that `src\data\recovery.py` is now inside the existing project.

No additional package installation is required. Keep the saved data available.

## Recover this download

Open PowerShell in your existing project:

```powershell
Set-Location 'D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project'
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\step3_collect_data.py --recover 'data\raw\mp_20260913T170343Z_40c2a1'
```

Paste your API key at the hidden prompt, then press Enter. The characters remain
invisible. The script validates saved counts and electrode page sequences before
creating a recovered snapshot. It does not download electrode records again.

The supplied console log totals 6,683 electrode records across eight ions.
The exact reused metadata count is determined from the saved file: valid records
may have been written after the last progress message.

```text
[OK] Reusing 6683 electrode records; no electrode download is needed.
[OK] Reusing ... exact-ID metadata records.
Retrying metadata for ... unresolved endpoint IDs...
...
STEP 3 DOWNLOAD AND INSPECTION COMPLETE
```

## Recovery behavior

- Preserves the original snapshot and creates a new folder ending in `_recovered`.
- Copies original electrode bytes and records source hashes.
- Reuses metadata that exactly matches requested endpoint IDs.
- Retries remaining IDs, including those missing from earlier batches.
- Retries unresolved batch results individually, with bounded response sizes.
- Logs unexpected IDs in `metadata_lookup_issues.jsonl` without assigning their
  symmetry to requested endpoints.
- Leaves unresolved symmetry fields missing and flags affected CSV rows.
- Generates the CSV tables and dataset report for review.
- Preserves partial recovery if a network, authentication, or schema error occurs.

The reference voltage labels and formulas are retained. Completion of collection
and inspection does not mean every endpoint has symmetry metadata or every row
is ready for training. Review `complete_basic_inputs`, row warnings, and the
report before selecting training inputs.

The old downloader did not persist its database version before this failure.
The report notes that database-version continuity between the original download
and recovery cannot be certified. Recovery-time versions do not date the original
electrode data. Original and recovery timestamps remain separate.

## Results to share

Upload the new report, whose exact path is printed in the terminal:

```text
reports\<new_snapshot_name_ending_in_recovered>\dataset_report.json
```

Diagnostics are at `data\raw\<new_snapshot>\metadata_lookup_issues.jsonl`.
Keep your API key private. Stop for dataset review before Step 4.

## Troubleshooting

| Message | Action |
|---|---|
| `unrecognized arguments: --recover` | Replace the old script with the repair's `scripts\step3_collect_data.py`. |
| `No module named src.data.recovery` | Copy the repair's `src` folder into the same existing project. |
| `manifest.json` not found | Check the snapshot folder name and current project directory. |
| Electrode counts or pages cannot be verified | Share the error and saved `manifest.json`. Metadata-only recovery cannot certify incomplete electrodes. |
| HTTP 401 or 403 | Check your API key and account access. |
| Recovery interrupted by a network error | Run `--recover` with the new incomplete recovery folder printed in the error output. |
| Unresolved symmetry IDs remain | Upload the report. Those fields are flagged as missing. |
| Database version changed | Share the error; known different versions are not merged as one consistent snapshot. |
| CSV permission error | Close the CSV in Excel and run offline inspection on the completed recovered snapshot. |

## Verification

41 offline tests passed in development, covering original behavior, unexpected
IDs, interrupted downloads, preserving source files, and missing-symmetry flags.
Tests use synthetic records, not research data. Live authenticated recovery must
run on your Windows computer with your key.

Optional local test command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_step3*.py" -v
```

The fresh-download command is also repaired. For this saved download use
`--recover` as above.
