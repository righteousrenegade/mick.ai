#!/bin/bash
echo "Setting up environment for AI Chat Interface..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies with specific NumPy version
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting AI Chat Interface..."
python run_chat_gui.py 