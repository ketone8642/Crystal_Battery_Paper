# Step 7: Connect the trained models to the prediction API

Step 6 completed on your Windows computer with run `step6_b8e6065bfc4e`.
All 17 training tests passed. The deprecation warnings in your log did not stop
the tests, final model fits or evaluation. No dependency change is needed for
those warnings at this stage.

## Review of your measured results

These values are copied from your uploaded console output, in volts:

| Training population | Family | Holdout MAE | Na MAE | K MAE |
|---|---|---:|---:|---:|
| Mixed ions | SVR | 0.6973 | 0.9115 | 1.3500 |
| Mixed ions | KRR | 0.8126 | 0.9973 | 1.4287 |
| Mixed ions | DNN | 0.6553 | 0.7431 | 1.2093 |
| Li only | SVR | 0.7802 | 0.9587 | 1.2083 |
| Li only | KRR | 0.8602 | 1.0010 | 1.2907 |
| Li only | DNN | 0.7872 | 0.9240 | 1.0039 |

Development cross-validation selected DNN for both populations. All six selected
candidates use the scaled features without PCA. The selected IDs are:

| Population | SVR | KRR | DNN |
|---|---|---|---|
| Mixed ions | `svr_scaled_1` | `krr_scaled_1` | `dnn_scaled_0` |
| Li only | `svr_scaled_1` | `krr_scaled_2` | `dnn_scaled_0` |

Li-only SVR has a slightly smaller holdout MAE than Li-only DNN, but the API keeps
the family choice made from CV. The two populations have different holdout sets;
their overall holdout scores do not directly compare performance on the same
lithium examples. Na/K scores are held-out-ion results from the same database.

The console output does not include the median baseline, per-ion holdout errors,
R2, or known/new-group Na/K errors. Those remain available in your local
`step6_report.json`; the API reads them rather than inventing missing values.

This remains a composition/symmetry adaptation, with grouped testing and a
current database snapshot. It is not an exact reproduction of the original paper.
The 14 priority voltage-label reviews remain unresolved. A computed prediction
does not certify an electrode's experimental performance or chemical stability.

## 1. Copy the addon into the project

Extract `battery_voltage_step7_api.zip` and copy its contents into:

```text
D:\2-1_Project School\Solution\battery_voltage_step2_windows\battery_voltage_project
```

Merge `app`, `src`, `scripts`, `tests`, `frontend` and `examples`. Replace the
supplied `app/main.py` and `scripts/check_environment.py` when prompted. The
new `app/main.py` activates the prediction routes. The environment checker now
accepts the updated health response.

The existing Step 4–6 source code, completed reports and six model bundles are
required. This addon supplies inference code; it does not contain your Windows
model weights or rerun training. No new packages or API key are required.

If an old Uvicorn server is running, stop it with Ctrl+C in its own terminal.

## 2. Run the API tests

In PowerShell from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_step7_api.py" -v
```

Expected ending:

```text
Ran 22 tests in ...s

OK
```

The tests train very small synthetic fixtures in a temporary directory to verify
actual TensorFlow/SVR/KRR serialization and HTTP inference. Their numerical
results are software tests, not battery research results. The fixtures are
removed after testing and the production loader rejects synthetic test runs.
These tests do not train on or modify your research data or saved models.

All 22 tests passed here with FastAPI 0.115.12, Pydantic 2.11.7, HTTPX 0.28.1,
TensorFlow 2.20.0, scikit-learn 1.6.1 and NumPy 2.1.3. Execution here used Linux
and Python 3.12; source was checked for Python 3.11 syntax. Your command above
verifies the Windows environment.

## 3. Configure your completed training run

```powershell
.\.venv\Scripts\python.exe scripts\step7_configure_api.py --step6-run step6_b8e6065bfc4e
```

This command:

1. Checks that Step 6 completed and its recorded output checksums still match.
2. Checks that the evaluated candidates match the frozen CV selection.
3. Loads all six model bundles and their original preprocessors.
4. Runs a local prediction through every model using the shared feature generator.
5. Saves the selected run in `config/active_model.json`.
6. Checks API readiness, a valid prediction and rejection of an invalid reaction.

Expected ending:

```text
[OK] All six bundles loaded and produced finite predictions
[OK] Readiness, prediction and invalid-input checks passed
Active run: step6_b8e6065bfc4e
...
STEP 7 API CONFIGURATION AND LOCAL CHECK PASSED
```

The command also prints a FePO4 → LiFePO4 prediction using the example's supplied
symmetry description. Its value comes from your saved model. It is an inference
check, not an independent performance benchmark.

## 4. Start the server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`, then keep this terminal open. The six
models are loaded once at application startup. Use one worker for this local
CPU setup; inference requests share the loaded bundles.

Open these pages in Chrome:

- [API status page](http://127.0.0.1:8000/)
- [Readiness](http://127.0.0.1:8000/ready)
- [Interactive API documentation](http://127.0.0.1:8000/docs)
- [Saved model information](http://127.0.0.1:8000/models)

`/ready` should return:

```json
{"ready": true, "run_id": "step6_b8e6065bfc4e", "loaded_models": 6}
```

`/health` reports server availability separately from model readiness. When no
run is configured, the server can be live while `/ready` and prediction endpoints
return HTTP 503. The status page tells you to configure the models.

The root page is a small API status page at this step. The full prediction form
and profile charts are Step 8.

## 5. Make a prediction in Chrome

In `/docs`:

1. Expand **POST /predict**.
2. Click **Try it out**.
3. Use the supplied example or paste the following JSON.
4. Click **Execute**.

```json
{
  "working_ion": "Li",
  "formula_charge": "FePO4",
  "formula_discharge": "LiFePO4",
  "crystal_system_charge": "orthorhombic",
  "spacegroup_number_charge": 62,
  "crystal_system_discharge": "orthorhombic",
  "spacegroup_number_discharge": 62,
  "model_family": "auto",
  "training_population": "auto"
}
```

Expected: HTTP **200**, with a numerical `predicted_voltage_V`, `run_id`, model
family/population, normalized concentration interval, saved evaluation context
and applicable notices. There is no fixed expected prediction number until
your own saved model is executed.

For this example, the response should identify the Li-only DNN and report
`fracA_charge=0`, `fracA_discharge=1/7` (approximately 0.142857), and working-ion
loading from 0 to 1 per reduced FePO4 host. It should report
`extrapolates_working_ion=false`.

The optional `fracA_charge` and `fracA_discharge` fields are atom fractions,
not the formula's insertion coefficient x. If provided, they must agree with
the formulas. Otherwise the API calculates them.

The example JSON is also provided at `examples/prediction_request.json`.

## 6. Optional PowerShell request

Open a second PowerShell window in the project root while Uvicorn keeps running:

```powershell
$predictionBody = Get-Content -Raw .\examples\prediction_request.json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $predictionBody | ConvertTo-Json -Depth 10
```

Download a prediction CSV from the supplied batch example:

```powershell
$batchBody = Get-Content -Raw .\examples\batch_request.json
Invoke-WebRequest -Uri 'http://127.0.0.1:8000/predict/batch.csv' -Method Post -ContentType 'application/json' -Body $batchBody -OutFile .\voltage_predictions.csv
```

## API routes

| Method and path | Purpose |
|---|---|
| `GET /health` | Server status, model-loaded flag and active run |
| `GET /ready` | HTTP 200 only when all six bundles are loaded |
| `GET /models` | Available models, default routing and saved evaluation results |
| `POST /predict` | One reaction |
| `POST /predict/batch` | 1–128 reactions, returned in their original order |
| `POST /predict/batch.csv` | The same batch request, returned as a downloadable CSV |
| `POST /predict/profile` | 2–50 contiguous reaction intervals |

A batch body uses `{"items": [reaction1, reaction2]}`. A profile body uses
`{"intervals": [reaction1, reaction2]}`. Every item follows the same prediction
schema. An invalid batch is rejected before any model is executed.

For a profile, all intervals must use the same working ion, reduced host and model
choices. Intervals must be supplied in increasing, contiguous loading order.
Shared endpoints must have matching symmetry. The response gives one average
voltage per interval and its loading limits; it does not interpolate a voltage
curve, invent missing structures or certify that intermediate phases exist.

## Model routing and interpretation

| Requested ion | Automatic training population | Automatic family for your run |
|---|---|---|
| Li, Na, K | Li only | DNN |
| Mg, Ca, Zn, Al, Y | Mixed ions | DNN |

The population rule follows the earlier application plan. The family comes from
`selection.json`, which was frozen using development CV. Routing is not changed
by comparing final holdout results. `model_family` can explicitly request `dnn`,
`svr` or `krr`. `training_population` can request `li_only` or `mixed_ions`;
Li-only routing is restricted to Li/Na/K in this application.

You can explicitly request mixed-ion predictions for Na/K for research comparison.
They remain transfer predictions because Na/K were absent from both training
populations. Do not reselect a model using these same held-out results and call
the resulting evaluation an untouched test.

`evaluation_reference` is read from your saved Step 6 report:

- Familiar ions: population-level grouped holdout metrics and the requested-ion
  subgroup if available.
- Na/K: their same-database held-out-ion metrics, with separate known/new-group
  results relative to the selected model's development set.

These metrics describe datasets. They are not uncertainty bounds for the current
input. The API does not claim whether your newly submitted host was in development;
the transfer subgroup results are reference statistics, not a host lookup.

The response also identifies inputs that change descriptors which were constant
during training. Those effects could not be learned by the saved model. This is
a limited support diagnostic, not calibrated out-of-distribution detection.
Negative predicted voltages are returned without clipping and flagged for review.

## Input validation and errors

- All seven material/reaction fields in the example are required.
- Ion symbols and crystal-system names must match the documented choices.
- Space-group numbers must be JSON integers from 1 to 230 and agree with the
  crystal system. A string such as `"62"` or a boolean is rejected.
- The formula parser supports integer stoichiometries and nested grouping. It
  does not silently interpret fractional occupancy, hydrates or isotope notation.
- Endpoint formulas must share the same non-working-ion host, and ion loading
  must increase from charge to discharge.
- Extra fields, including target voltage, source identifiers and review flags,
  are rejected rather than treated as predictors.
- Each formula is limited to 256 characters. Nonfinite numeric inputs are rejected.

| Symptom | Action |
|---|---|
| HTTP 422 | Read the `detail` messages and correct the input fields |
| HTTP 503 | Run the configuration command, check the startup terminal and restart Uvicorn |
| HTTP 500 during prediction | Share the server error; the API never supplies a substitute voltage |
| File integrity failure | Share the error. Do not edit checksum files or bypass the check |
| Wrong scikit-learn version | Use the existing project `.venv`, matching the model's saved environment |
| `No module named app.schemas` or `src.inference.registry` | Merge every supplied addon folder |
| Port 8000 already occupied | Stop the old server or run with `--port 8001` and use that port in Chrome |
| `/docs` does not show the new routes | Restart Uvicorn from the correct project after copying the addon |
| Swagger page cannot load its assets | Use the PowerShell request; the standard Swagger UI loads assets from a CDN |

Configuration changes take effect when the server restarts. Changing configuration
does not retrain models. Model bundles should come from your trusted local training
run, because the saved kernel estimators use joblib serialization.

The server binds to `127.0.0.1` for local use. Hosting, authentication and shared
network deployment are outside this step.

## Source files in this addon

| File | Responsibility |
|---|---|
| `app/main.py` | Startup, status, model information and prediction routes |
| `app/schemas.py` | Typed requests, reaction validation and response documentation |
| `src/inference/registry.py` | Completed-run verification, model loading, routing and inference |
| `scripts/step7_configure_api.py` | Configure your run and check live inference locally |
| `scripts/check_environment.py` | Existing checker updated for the prediction-stage health response |
| `tests/test_step7_api.py` | 22 API and model-loading tests |
| `frontend/api_status.html` | Minimal status page with links to API documentation |
| `examples/*.json` | Runnable single and batch requests |

The startup pattern follows [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/).
Request checks use [Pydantic validators](https://pydantic.dev/docs/validation/2.11/concepts/validators/)
and the same reaction feature generator used during training.

Share the configuration command's output and one successful `/predict` response.
Then proceed to Step 8: the browser prediction form, voltage profiles and downloads.
