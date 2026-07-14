@echo off
REM Run the pronunciation grid (Stage 1).
REM Other scripts in this folder can be run by name, e.g.:
REM   run.bat                       -> pronunciation_test.py
REM   run.bat build_backgrounds     -> build_backgrounds.py [args...]
REM   run.bat filter_rirs           -> filter_rirs.py [args...]

if not defined VIRTUAL_ENV (
    call "%~dp0.venv\Scripts\activate.bat"
)

set "SCRIPT=%~1"
if "%SCRIPT%"=="" (
    python "%~dp0pronunciation_test.py"
) else (
    shift
    python "%~dp0%SCRIPT%.py" %*
)
