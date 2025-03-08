@echo off
echo Setting up environment for AI Chat Interface...

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies with specific NumPy version
echo Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo Starting AI Chat Interface...
python run_chat_gui.py

pause 