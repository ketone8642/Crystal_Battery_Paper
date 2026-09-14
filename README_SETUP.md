# Step 2: Windows 11 and Chrome

This is the setup stage of a paper-based battery electrode voltage project.
It contains pinned direct dependencies, the planned folder structure, a runtime
checker, and a small FastAPI page for checking Chrome connectivity. There is no
dataset, trained model, or voltage prediction implementation yet.

The paper's full 237-feature specification and several training settings are
absent from the uploaded PDF. Later implementation choices will be documented.

## 1. Python and editor

Use Python 3.11, 64-bit x86-64. First check in Windows Command Prompt:

```bat
py -3.11 --version
py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
```

Expected: Python 3.11.x, then 64.

If Python 3.11 is missing, the Python 3.11.9 release page provides a traditional
Windows installer (64-bit): https://www.python.org/downloads/release/python-3119/
This is an older installer for the 3.11 line, not the latest Python patch release.
Enable the Python launcher and Add python.exe to PATH during installation.
Keep another installed Python version if you need it for other projects.

Install VS Code if needed: https://code.visualstudio.com/download
In Extensions, install Python by Microsoft.

## 2. Extract the project

Extract the ZIP so that the project root is, for example:

```text
C:\Projects\battery_voltage_project
```

The requirements.txt file must be directly inside that folder. Avoid a second,
accidentally nested battery_voltage_project folder. You may use another writable
location and adjust the cd command.

All commands below are for Command Prompt (cmd.exe).

```bat
cd /d C:\Projects\battery_voltage_project
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Expected final dependency check: No broken requirements found.
Wait for the installation command to finish before running the next command.
The TensorFlow package is large, and dependencies add to the download size.
Materials Project and featurization packages will be added in the data stage.

## 3. Verify the environment

```bat
python scripts\check_environment.py
```

Success ends with:

```text
STEP 2 ENVIRONMENT CHECK PASSED
Next: start the server and verify the page in Chrome.
```

The TensorFlow check only adds three numbers; it does not train a model.
After success, record all resolved versions:

```bat
python -m pip freeze > requirements-lock.txt
```

This records the transitive dependencies for this Windows environment. It is not
a cross-platform lockfile. Regenerate it after adding data-stage dependencies.

## 4. Verify Chrome

```bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep the terminal open. In Chrome visit:

- http://127.0.0.1:8000 : setup page
- http://127.0.0.1:8000/health : server status JSON
- http://127.0.0.1:8000/docs : interactive API documentation

The page should show: Server connection successful. Model training is pending.

The health response is:

```json
{"status":"ok","stage":"environment_setup","model_loaded":false}
```

This confirms server connectivity only. Model validation comes in later steps.
Use Ctrl+C in the terminal to stop the server.

## 5. VS Code interpreter

Open the project folder in VS Code. Press Ctrl+Shift+P, run Python: Select
Interpreter, and select .venv\Scripts\python.exe under this project.
Choose Command Prompt for the integrated terminal when following these commands.

## Folder responsibilities

| Path | Purpose |
| --- | --- |
| app/ | API application; currently setup routes only |
| frontend/ | Browser page; currently setup page only |
| scripts/ | Environment checks and later command-line utilities |
| src/data/ | Future retrieval and validation code |
| src/features/ | Future shared descriptor generation |
| src/training/ | Future DNN, SVR, KRR training |
| src/inference/ | Future saved-model loading and prediction |
| data/raw/ | Original downloaded records |
| data/processed/ | Cleaned reaction data |
| data/splits/ | Saved train/test group assignments |
| models/li_only/ | Future lithium-trained model bundles |
| models/all_ions/ | Future mixed-ion model bundles |
| reports/figures/ | Future evaluation plots |
| tests/ | Future application and data validation tests |

## Troubleshooting

| Problem | Action |
| --- | --- |
| py is not recognized | Install/enable the Python launcher, then reopen Command Prompt. |
| No suitable Python runtime found | Install Python 3.11 x86-64; another major/minor version does not satisfy py -3.11. |
| requirements.txt is not found | Change to the extracted project root and use dir to confirm the file exists. |
| PowerShell script execution error | Use Command Prompt and activate.bat as instructed. |
| TensorFlow DLL load error | Check 64-bit Python and install/repair the current Microsoft Visual C++ x64 Redistributable, then restart the terminal. |
| ModuleNotFoundError | Run python -m pip --version and check that its path is inside the project's .venv; reinstall requirements there. |
| Dependency resolution error | Stop and share the final resolver message; do not force installation with --no-deps. |
| No matching TensorFlow distribution | Check Python 3.11, 64-bit, an up-to-date pip, and access to the package index. |
| Chrome connection refused | Start Uvicorn from the project root and leave it running; use http rather than https. |
| WinError 10048 / address in use | Choose port 8001 in the command and in Chrome. |
| No GPU detected | Expected for this native-Windows TensorFlow CPU setup. |

## Validation status

This package was authored in a Linux workspace. Python syntax and available
numerical components can be checked there, but the complete pinned environment
and native Windows DLL loading must be verified on your Windows computer.
The environment checker performs those checks without an API key or dataset.

## Official references

- Python virtual environments: https://docs.python.org/3.11/library/venv.html
- TensorFlow Windows CPU installation: https://www.tensorflow.org/install/pip
- TensorFlow 2.20.0 Windows wheels: https://pypi.org/project/tensorflow/2.20.0/
- FastAPI first steps: https://fastapi.tiangolo.com/tutorial/first-steps/
- VS Code interpreter selection: https://code.visualstudio.com/docs/python/environments
- Microsoft Visual C++ Redistributable: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

Stop after Step 2 and report the environment-check result and Chrome page status.
Step 3 will cover data access and preparation after confirmation.
