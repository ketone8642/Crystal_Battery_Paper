# Step 8: Browser interface

Your Step 7 API returned HTTP 200 and predicted 3.4650564309719782 V for the
FePO4 -> LiFePO4 example with your Li-only DNN. This addon connects a browser
workspace to that API and your six saved models.

## 1. Install the addon

Stop the running Uvicorn server with **Ctrl+C** in its terminal.

Extract `battery_voltage_step8_interface.zip`. Copy the enclosed files and
folders directly into your existing project folder:

```text
D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project
```

Merge `app`, `src`, `frontend`, `scripts` and `tests`, and replace the files
supplied by this addon. After copying, check that these files exist:

```text
frontend\assets\app.js
frontend\assets\styles.css
scripts\step8_check_ui.py
```

The addon requires your completed Step 7 installation. It updates the homepage,
static asset serving and model catalog, and includes the corresponding tests.
It does not contain model weights or a replacement active-model configuration.
No new packages, API key, Node.js installation or model retraining are required
to run the application.

## 2. Check your installed interface and models

Open PowerShell and run these commands separately:

```powershell
Set-Location 'D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project'
```

```powershell
.\.venv\Scripts\python.exe scripts\step8_check_ui.py
```

Expected ending:

```text
[OK] All six configured model bundles are ready
[OK] Four workspace sections and local browser assets are available
[OK] JavaScript files are served with module-compatible content types
[OK] FePO4 -> LiFePO4 prediction: 3.4651 V
STEP 8 UI ASSET AND API CHECK PASSED
```

The prediction is calculated by your configured model. The number above is
based on your reported Step 7 output, not a value embedded in the interface.
The checker loads the models, serves the assets through FastAPI's test client,
and calls the prediction route. It does not execute Chrome.

## 3. Start the application

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Open **http://127.0.0.1:8000** in Chrome and press
**Ctrl+F5** to refresh the application files. The status should read
**6 models ready**. Open the page through Uvicorn; double-clicking the HTML file
does not provide the API or module routes.

## 4. Verify the four workflows

### Single prediction

Click **Load FePO4 example**, leave both model settings on Automatic, and click
**Predict voltage**. The example uses:

| Field | Charged endpoint | Discharged endpoint |
|---|---|---|
| Working ion | Li | Li |
| Formula | FePO4 | LiFePO4 |
| Space-group number | 62 | 62 |
| Derived crystal system | Orthorhombic | Orthorhombic |

For your current run, expect approximately **3.465 V**. The card identifies
the Li-only DNN, reports the insertion interval 0 -> 1 ions per reduced host,
and shows saved evaluation context. This is a functional inference example,
not an independent accuracy test.

Download the CSV or JSON. The display rounds to three decimal places; downloads
retain the API's full numerical precision and model/run identifiers. Editing
an input clears the previous output so it cannot be mistaken for a new result.

The app calculates working-ion atom fractions and reduced-host loadings from
the endpoint formulas. The discharged Li atom fraction here is 1/7, while its
Li loading per FePO4 host is 1. These quantities use different denominators.

To check error handling, change only the discharged formula to `LiCoO2` and
submit. The API should reject the inconsistent host. Reload the example to
restore the valid input.

### Batch predictions

1. Open **Batch predictions** and click **Download input template**.
2. Upload that CSV using **Choose a CSV file**.
3. Click **Predict batch**. The two template rows request the same Li reaction
   using Automatic and SVR model choices.
4. Download the results CSV and check the reaction, voltage, model and run columns.

The limit is 128 data rows and 1 MB per input file. Keep the template headers.
Required fields are:

```text
working_ion,formula_charge,formula_discharge,crystal_system_charge,crystal_system_discharge,spacegroup_number_charge,spacegroup_number_discharge
```

Optional fields are `fracA_charge`, `fracA_discharge`, `model_family` and
`training_population`. Blank optional values are omitted. Families are
`auto`, `dnn`, `svr`, `krr`; populations are `auto`, `li_only`, `mixed_ions`.
Unlike the browser's individual-entry forms, CSV inputs include crystal-system
columns; they must agree with each endpoint's supplied space group.

Training tables contain additional targets/provenance columns and are not
direct prediction input files. Copy the required input columns into the
template. A malformed or invalid reaction causes the request to fail; it does
not produce a made-up prediction or silently skip the row.

### Voltage profile

Enter 3 to 51 physically justified endpoint states for one host, ordered by
increasing working-ion loading. Supply the known space-group number for each
state. The app makes 2 to 50 adjacent interval requests using the selected ion
and common model settings.

Use endpoint states from a suitable multi-step electrode record in your
downloaded dataset. Do not invent an extra intercalation phase merely to make
a chart. This workflow accepts integer-coefficient formulas under the existing
API's composition rules.

The chart shows a horizontal segment across each interval's loading limits,
using the returned average voltage. Dashed joins are visual guides. The table
and downloadable CSV provide the interval boundaries and individual predictions.
The graph is an approximate predicted profile; its shape is not forced to
decrease, and negative predictions are not clipped.

### Model results

The page reads your active training report through `/models`. It shows all six
saved models, development CV MAE, holdout MAE, held-out Na/K MAE and available
median-baseline results. CV-choice badges identify the family selected during
development. The page does not select a family from final test results.

Automatic ion routing remains Li/Na/K -> Li-only, and Mg/Ca/Zn/Al/Y -> mixed ions.
Explicit model/population controls allow comparison without changing that
default. Na/K are absent from both training populations. Their displayed
evaluation results are held-out-ion tests from the same database; some host
groups can be familiar from development data.

## Reading the reported errors

MAE is an average error over the named dataset. It is not an uncertainty bound
for the current prediction. Transfer notices and changed training-constant
descriptor notices come from the API and are preserved in downloads.

The system uses computed Materials Project labels and 740 composition/symmetry
features. It is an adaptation of the paper, and does not reconstruct the
paper's missing original descriptor specification. It does not establish phase
stability or experimental performance. The earlier priority label reviews
remain a research-data limitation.

## Troubleshooting

| Symptom | Action |
|---|---|
| The previous setup/status page appears | Stop and restart Uvicorn from this project folder, then press Ctrl+F5 in Chrome. |
| Unstyled page or no button response | Confirm all files under `frontend/assets` were copied; run the Step 8 checker. |
| A JavaScript module content-type error appears | Replace `app/main.py` from this addon and restart Uvicorn. It registers `.js` and `.mjs` MIME types explicitly for Windows. |
| Models are not ready | Read the terminal error, confirm Step 7 configuration and saved models exist, and use Reconnect after resolving it. |
| Active configuration is missing | Run the existing Step 7 configuration command for `step6_b8e6065bfc4e`; do not retrain. |
| Chrome reports connection refused | Keep the Uvicorn terminal open and use the same host/port printed there. |
| Port 8000 is occupied | Stop the old server, or run with `--port 8001` and open `http://127.0.0.1:8001`. |
| A chemistry, symmetry or profile error appears | Correct the indicated field or CSV row; verify shared host, valid space group and increasing contiguous loading. |
| Prediction differs from the earlier example | Check the exact endpoint formulas, space groups, chosen family/population and active run. Different inputs/models can give different results. |

## Developer verification already performed

All of the following passed in the Linux verification environment:

- 22 API regression tests using temporary synthetic model bundles.
- 12 JavaScript tests covering CSV parsing, export precision, input checks,
  error rendering and profile geometry.
- 16 headless Chromium 153 browser checks covering live API inference, model
  identity, invalid inputs, stale-response handling, downloads, batch upload,
  profile segments, model catalog, keyboard navigation, mobile layout and
  unavailable-model handling. Desktop/profile/mobile screenshots were inspected.

These are software checks with explicitly labelled synthetic models. Your
Windows research-model files were not available here. The check command and
Chrome actions above verify the installed interface with your actual models.
Logs and the browser check list are included under `verification/step8`.

Optional developer commands, if Node.js and Playwright are already installed:

```text
node --test tests/test_step8_ui.mjs
python tests/serve_step8_fixture.py
```

Run `node tests/test_step8_browser.cjs` in a separate terminal while the fixture
server is running on port 8768. It deliberately verifies the synthetic test
banner; do not point it at your production research server. Optional environment
variables are `BATTERY_TEST_URL`, `BATTERY_TEST_OUTPUT` and
`BATTERY_TEST_CHROMIUM` (an installed Chromium executable). The temporary test
models do not replace production models. Stop the fixture server with Ctrl+C.

**Stop after this step. Share the checker output and the Chrome prediction
screen. After Step 8 works on your computer, Step 9 covers final application
verification and the consolidated run/demo documentation.**
