@echo off
echo Fixing NumPy compatibility issue...

REM Activate virtual environment if it exists
if exist venv (
    call venv\Scripts\activate.bat
)

REM Uninstall NumPy 2.x
echo Uninstalling NumPy...
pip uninstall -y numpy

REM Install NumPy 1.x
echo Installing NumPy 1.x...
pip install "numpy<2.0.0"

echo NumPy fixed! You can now run the application with:
echo python run_chat_gui.py

pause 