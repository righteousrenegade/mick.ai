#!/bin/bash
echo "Fixing NumPy compatibility issue..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Uninstall NumPy 2.x
echo "Uninstalling NumPy..."
pip uninstall -y numpy

# Install NumPy 1.x
echo "Installing NumPy 1.x..."
pip install "numpy<2.0.0"

echo "NumPy fixed! You can now run the application with:"
echo "python run_chat_gui.py" 