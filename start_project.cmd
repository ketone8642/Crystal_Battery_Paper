@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project virtual environment is missing. Complete the Python setup first.
  pause
  exit /b 1
)
echo Starting Battery Electrode Voltage Prediction...
echo After Application startup complete, open http://127.0.0.1:8000 in Chrome.
echo Keep this window open. Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
endlocal
