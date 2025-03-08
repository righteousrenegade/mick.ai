# Mick.AI

Mick.AI is a collection of useful python scripts to do various things with AI, like local document search or synthetic data generation or custom model training.

Mick.AI is named after Mickey Goldmill, from the Rocky movies. He was a legendary trainer helping to make a fine-tuned machine. Thus, the name of this repo.

Features/Capabilities for now will be separated into top-level directories. Each feature will have its own README to explain how to use it.

See the [Trainer README](train/README.md) for instructions on how to install and use the trainer scripts.

# Script Launcher

A GUI application for easily running scripts from your workspace.

## Features

- Discover and run Python, Bash, PowerShell, and Batch scripts
- View script descriptions and details
- Real-time output display
- Ability to stop running scripts
- Set custom scripts directory

## Requirements

- Python 3.6 or higher
- Tkinter (usually comes with Python)

## Installation

No installation is required. Simply run the script:

```
python script_launcher.py
```

## Usage

1. Launch the application by running `script_launcher.py`
2. The application will automatically scan the current directory for scripts
3. Click "Set Scripts Directory" to choose a different directory
4. Select a script from the list to view its details
5. Click "Run Script" or double-click on a script to execute it
6. View the output in real-time in the output panel
7. Click "Stop Script" to terminate a running script
8. Click "Clear Output" to clear the output panel
9. Click "Refresh Scripts" to update the script list

## Supported Script Types

- Python scripts (.py)
- Bash scripts (.sh)
- PowerShell scripts (.ps1)
- Batch files (.bat, .cmd)

## Notes

- Script descriptions are automatically extracted from comments at the beginning of the script
- The application runs scripts in a separate thread, so the UI remains responsive
- Output is captured and displayed in real-time