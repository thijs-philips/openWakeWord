@echo off
REM Creates a venv inside this folder and installs the demo dependencies.
REM Run once after cloning. Re-run safely; will skip what already exists.

setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt --extra-index-url https://pypi.org/simple || goto :error

echo Pre-fetching openwakeword preprocessor models (melspec + speech embedding)...
".venv\Scripts\python.exe" -c "import openwakeword; openwakeword.utils.download_models()" || goto :error

echo.
echo Done. Try:  .venv\Scripts\python.exe mic_demo.py --list-devices
endlocal
exit /b 0

:error
echo.
echo Setup FAILED. See error above.
endlocal
exit /b 1
