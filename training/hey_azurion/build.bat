@echo off
REM Build the local virtual environment for hey_azurion training utilities.
echo === Building hey_azurion training environment ===

if not exist "%~dp0.venv" (
    echo Creating virtual environment...
    python -m venv "%~dp0.venv"
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call "%~dp0.venv\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r "%~dp0requirements.txt"

echo === Build complete ===
