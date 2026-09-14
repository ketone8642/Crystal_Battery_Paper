# Step 3 — Download and inspect battery voltage data

**Recovery update:** If your electrode download finished but symmetry metadata
retrieval failed, follow `README_STEP3_RECOVERY.md` and use `--recover PATH` to
reuse the saved records. The old advice below to restart applies only when the
electrode download itself is incomplete. Metadata queries now retain exact ID
matches, retry unresolved IDs, and log unmatched responses for review.

This addon extends your working Step 2 project on Windows 11. It downloads
computed insertion-electrode records from Materials Project and produces CSV
tables plus a data-quality report. It does not train a model.

The source code is included. No additional packages are required: this stage
uses Python's standard library. Continue using your existing Python 3.11 virtual
environment. Chrome is used to obtain your API key; the download runs in
PowerShell and does not require the FastAPI server to be running.

## Important research distinction

The paper used a historical dataset. These scripts retrieve the current
Materials Project database, so the records and their number can differ from
the paper's 3,977 cleaned samples. This is a documented adaptation, not an exact
reproduction of its original dataset or results.

The downloaded voltages are computational predictions provided by Materials
Project, not experimental measurements. The download preserves the source
labels; it never invents a missing voltage or generates demonstration data as
research data. Synthetic records exist only inside the offline software tests.

## 1. Add these files to your existing project

Extract `battery_voltage_step3_data_addon.zip` into a temporary folder. Inside
the extracted folder, select `scripts`, `src`, `tests`, `README_STEP3.md`, and
`DATA_DICTIONARY_STEP3.md`. Copy them into:

```text
D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project
```

Merge the folders. The new script must end up at
`battery_voltage_project\scripts\step3_collect_data.py`, not inside another
`battery_voltage_project` folder. The archive contains no Step 2 application or
dependency files to replace.

Open PowerShell in the project folder. If your existing terminal is displaying
server logs, open another PowerShell window or stop that server with Ctrl+C.

```powershell
Set-Location 'D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project'
Test-Path .\scripts\step3_collect_data.py
.\.venv\Scripts\python.exe --version
```

Expected: `True` and `Python 3.11.9` on your machine.

## 2. Run the offline tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_step3_data.py" -v
```

Expected ending:

```text
Ran 25 tests in ...s

OK
```

These tests use a fake HTTP service and temporary directories, require no API
key or internet, and do not add test records to your project's data folders.
They check authentication error handling, retries, pagination completeness,
database-version changes, missing targets, zero/negative voltages, duplicate
and conflicting intervals, snapshot integrity, and the complete export flow.

Validation performed before delivery: all 25 tests passed on Linux with Python
3.12; the source also passed Python 3.11 syntax parsing. A live authenticated
download and execution on your Windows machine remain to be verified locally.

## 3. Obtain your Materials Project API key

Open the [Materials Project dashboard](https://next-gen.materialsproject.org/dashboard)
in Chrome, sign in, and copy your API key. See the
[official API getting-started guide](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started).

The scripts ask for the key with hidden input. Paste it into that local prompt
and press Enter; characters are not displayed. Do not paste the key into this
chat. It is not written into source files, reports, or download manifests.

If `MP_API_KEY` already exists in your terminal environment, the script reads
that value instead and does not prompt. A fresh hidden prompt is otherwise
shown each time a download/check command starts.

## 4. Check API access with one record

```powershell
.\.venv\Scripts\python.exe scripts\step3_collect_data.py --check --ions Li
```

When prompted, paste your key. A successful response prints:

```text
[OK] Materials Project API request succeeded.
API-reported Li electrode count: <actual count>
Returned fields: <fields actually returned>
Adjacent intervals in this first record: <actual count>
API CHECK COMPLETE. No dataset has been downloaded or model trained in this check.
```

The bracketed values above are placeholders. The script prints actual values.
If the command reports an error or no records, address that before the full
download; see troubleshooting below.

## 5. Download and inspect the dataset

```powershell
.\.venv\Scripts\python.exe scripts\step3_collect_data.py
```

Paste your key again if prompted. The default ion list is:

```text
Li Na K Mg Ca Zn Al Y
```

Li, Mg, Ca, Zn, Al, and Y correspond to the paper's main training-ion selection.
Na and K are extra records for later evaluation or extension. Downloading them
does not automatically add them to a training set. No split is made here.

The script requests pages of electrode records, then requests symmetry metadata
for charged/discharged endpoint materials, including intermediate endpoints.
Progress is printed after each page or metadata batch. Total time depends on
the server, number of records, rate limits, and your internet connection.

Every run creates a new timestamped folder. Re-running does not overwrite an
earlier raw snapshot and does not resume an incomplete download. For a smaller
initial download, you can explicitly limit the ion list:

```powershell
.\.venv\Scripts\python.exe scripts\step3_collect_data.py --ions Li
```

For smaller electrode responses:

```powershell
.\.venv\Scripts\python.exe scripts\step3_collect_data.py --page-size 50
```

## 6. Review the output

Successful completion ends with:

```text
STEP 3 DATASET INSPECTION
Snapshot: mp_<UTC timestamp>_<unique suffix>
Unique electrode documents: <actual count>
Adjacent voltage intervals: <actual count>
Intervals passing basic checks: <actual count>
Intervals with complete basic inputs: <actual count>
...
STEP 3 DOWNLOAD AND INSPECTION COMPLETE
Share dataset_report.json for review before Step 4.
```

The script prints the exact paths. For each snapshot:

| Location | Purpose |
| --- | --- |
| `data/raw/<snapshot>/electrodes.jsonl` | Original projected electrode API records; one JSON object per line |
| `data/raw/<snapshot>/materials.jsonl` | Retrieved endpoint material metadata |
| `data/raw/<snapshot>/manifest.json` | Download status, timestamps, selected fields, page counts, database version when supplied, and SHA-256 file checksums |
| `data/processed/<snapshot>/electrode_summary.csv` | One row per distinct electrode document: voltage across its overall composition range |
| `data/processed/<snapshot>/voltage_intervals.csv` | One row per provided adjacent voltage interval |
| `data/processed/<snapshot>/intervals_for_review.csv` | Adjacent rows with issues or warnings |
| `reports/<snapshot>/dataset_report.json` | Detailed machine-readable counts, voltage statistics, issues, and limitations |
| `reports/<snapshot>/dataset_report.txt` | Short human-readable summary |

These are projected records: the requested fields are preserved, not every
possible field in the complete MP database. Missing metadata is listed in the
report. Inspecting data does not certify that it is ready for model training.

**Do not concatenate the summary and interval tables.** For example, an
electrode with two insertion stages has one overall voltage and two interval
voltages. Treating all three as independent observations can overweight the
same reaction family. We retain both representations to choose the appropriate
training unit after inspecting your report.

The inspector collapses identical electrode documents and counts them. It
flags repeated interval identities and conflicting labels while retaining
their rows for review. Rows without valid labels stay visible; their voltage
cells are blank. Real numeric zero and negative voltages are retained.

To rebuild the tables from the latest complete local snapshot without internet
or an API key:

```powershell
.\.venv\Scripts\python.exe scripts\step3_inspect_data.py
```

To choose a specific snapshot, replace the example name with your actual folder:

```powershell
.\.venv\Scripts\python.exe scripts\step3_inspect_data.py --snapshot 'data\raw\mp_YOUR_ACTUAL_SNAPSHOT'
```

## Architecture and source files

| Component | Responsibility |
| --- | --- |
| `scripts/step3_collect_data.py` | CLI, hidden key prompt, download orchestration, snapshot creation, automatic inspection |
| `src/data/mp_download.py` | Official REST requests, bounded retries, pagination checks, endpoint-ID collection |
| `src/data/inspection.py` | Separate overall/adjacent tables, basic validation, duplicate/conflict flags, CSV and report export |
| `scripts/step3_inspect_data.py` | Offline inspection of an existing complete snapshot |
| `tests/test_step3_data.py` | Offline behavior tests with explicitly invented fixtures |
| `DATA_DICTIONARY_STEP3.md` | Field meanings, examples, and limits of validation |

The downloader sends read-only HTTPS requests to
`https://api.materialsproject.org/materials/insertion_electrodes/` and
`https://api.materialsproject.org/materials/summary/` using the `X-API-KEY`
header. Pagination uses `_limit`/`_skip` and verifies `meta.total_doc`.
Changed totals, repeated full pages, missing totals, and inconsistent reported
database versions stop the download. These checks catch common inconsistencies;
they do not provide an atomic database snapshot if a server changes records
without changing its reported version or total.

The original JSONL files are checksum-verified before export. Incomplete
downloads are marked failed/interrupted and cannot be exported as completed
datasets. The server setup and browser page from Step 2 remain separate from
this command-line data workflow.

## Common errors and solutions

| Error or symptom | Action |
| --- | --- |
| `can't open file ... step3_collect_data.py` | Merge the ZIP contents into the correct project folder. `Test-Path .\scripts\step3_collect_data.py` should print `True`. |
| `.venv\Scripts\python.exe` not found | Return to the existing Step 2 project directory containing your working virtual environment. |
| Nothing appears while pasting the key | Hidden input is expected. Paste once, then press Enter. |
| HTTP 401 or 403 | Recopy the key from your Materials Project dashboard and check account access. If an old `MP_API_KEY` is set, remove it from this PowerShell session with `Remove-Item Env:MP_API_KEY -ErrorAction SilentlyContinue`, then rerun for a hidden prompt. |
| HTTP 422 | The service rejected a field/filter. Share the script's error output without your key so the current API schema can be checked. Do not invent replacement field names. |
| HTTP 404/410 or redirect error | The route may have changed. Share the error so the official API route can be checked. |
| HTTP 429/500/502/503/504 | The script retries up to three attempts. If retries are exhausted, rerun later; a new snapshot is created. |
| Connection timeout, proxy, or certificate error | Check your internet connection and your institution's proxy/certificate configuration. Ask IT if access to `api.materialsproject.org` is blocked. |
| Response exceeds 64 MiB | Retry using `--page-size 50`. |
| `KeyboardInterrupt` / Ctrl+C | The raw partial download is preserved and marked interrupted. Rerun when ready; it will start a new snapshot. |
| Snapshot integrity check failed | Raw files have changed since download. Keep them for investigation and download a fresh snapshot. |
| No adjacent intervals or no complete basic inputs | Share the report. The software does not fabricate endpoint information or labels to proceed. |
| Permission denied on a CSV during offline inspection | Close the CSV in Excel and run the inspection command again. |

## Stop after this step

Upload `reports/<snapshot>/dataset_report.json` or paste the final console
summary. Do not upload your API key. We will review the actual data before
Step 4: chemical/provenance validation, duplicate resolution, grouped splitting,
and feature preparation. Model training follows later.

Official implementation references checked for this addon:

- [Materials Project API getting started](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started)
- [Official electrode API client](https://github.com/materialsproject/api/blob/main/mp_api/client/routes/materials/electrodes.py)
- [Official API pagination client](https://github.com/materialsproject/api/blob/main/mp_api/client/core/client.py)
- [Emmet electrode document definitions](https://github.com/materialsproject/emmet/blob/main/emmet-core/emmet/core/electrode.py)
