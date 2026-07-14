@echo off
REM Launches the microphone wakeword tester.
REM Usage: run_mic_test.bat [extra args, e.g. --threshold 0.7]

setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=D:\Github\openWakeWord"
set "PYTHONIOENCODING=utf-8"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%mic_test.py" %*
endlocal
