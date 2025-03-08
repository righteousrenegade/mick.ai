#!/usr/bin/env python3
"""
Master GUI for JAImes Madison AI - Provides a unified interface for:
- Processing Federalist Papers data
- Cleaning training data
- Training the model
- Chatting with the trained model
"""

import sys
import os
import threading
import time
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
                            QProgressBar, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor

# Import the modules we need
try:
    from chat_gui import AIModelChat
except ImportError:
    print("Warning: chat_gui module not found. Chat functionality will be disabled.")
    AIModelChat = None

# Function to redirect stdout to a QTextEdit
class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            self.text_widget.append(self.buffer)
            self.buffer = ""
            # Scroll to the bottom
            cursor = self.text_widget.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_widget.setTextCursor(cursor)
        
    def flush(self):
        if self.buffer:
            self.text_widget.append(self.buffer)
            self.buffer = ""

# Worker thread for running processes
class ProcessWorker(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, process_type, params=None):
        super().__init__()
        self.process_type = process_type
        self.params = params or {}
        self.running = True
    
    def run(self):
        try:
            if self.process_type == "process_data":
                self.run_process_data()
            elif self.process_type == "clean_data":
                self.run_clean_data()
            elif self.process_type == "train_model":
                self.run_train_model()
            self.finished_signal.emit(True, f"{self.process_type} completed successfully!")
        except Exception as e:
            self.finished_signal.emit(False, f"Error in {self.process_type}: {str(e)}")
    
    def run_process_data(self):
        # Import and run the processdata module
        try:
            from processdata import test_api_connection, chunk_text, process_chunk, main as m
            # Redirect stdout to our signal
            original_stdout = sys.stdout
            sys.stdout = self
            m()
            sys.stdout = original_stdout
        except ImportError as e:
            self.update_signal.emit(f"Error: processdata.py module not found. {e}")
    
    def run_clean_data(self):
        # Import and run the cleandata module
        try:
            from cleandata import clean_qa_pairs
            # Redirect stdout to our signal
            original_stdout = sys.stdout
            sys.stdout = self
            input_file = self.params.get('input_file', 'trainingdata2.txt')
            output_file = self.params.get('output_file', 'cleaned_trainingdata.txt')
            clean_qa_pairs(input_file, output_file)
            sys.stdout = original_stdout
        except ImportError:
            self.update_signal.emit("Error: cleandata.py module not found.")
    
    def run_train_model(self):
        # Import and run the train module
        try:
            from train import train_model
            # Redirect stdout to our signal
            original_stdout = sys.stdout
            sys.stdout = self
            model_name = self.params.get('model_name', 'gpt2-medium')
            num_epochs = self.params.get('num_epochs', 5)
            batch_size = self.params.get('batch_size', 4)
            learning_rate = self.params.get('learning_rate', 5e-6)
            display_qa_pairs = self.params.get('display_qa_pairs', False)
            use_high_quality = self.params.get('use_high_quality', True)
            
            train_model(
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                display_qa_pairs=display_qa_pairs,
                use_high_quality=use_high_quality
            )
            sys.stdout = original_stdout
        except ImportError:
            self.update_signal.emit("Error: train.py module not found.")
    
    def write(self, text):
        # This allows us to capture print statements from the imported modules
        self.update_signal.emit(text)
    
    def flush(self):
        pass
    
    def stop(self):
        self.running = False

class MasterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JAImes Madison AI - Master Control")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create the main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs for each functionality
        self.create_process_tab()
        self.create_clean_tab()
        self.create_train_tab()
        self.create_chat_tab()
        
        # Initialize worker thread
        self.worker = None
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        # Set the font for the entire application
        font = QFont()
        font.setPointSize(10)
        QApplication.setFont(font)
        
        # Set the style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 2px solid #4a86e8;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 24px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
    
    def create_process_tab(self):
        # Create the Process Data tab
        process_tab = QWidget()
        layout = QVBoxLayout()
        
        # Add description
        description = QLabel("Process Federalist Papers data to create training examples.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        controls_layout.addWidget(self.process_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Add output area
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        self.process_output = QTextEdit()
        self.process_output.setReadOnly(True)
        output_layout.addWidget(self.process_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        process_tab.setLayout(layout)
        self.tabs.addTab(process_tab, "Process Data")
    
    def create_clean_tab(self):
        # Create the Clean Data tab
        clean_tab = QWidget()
        layout = QVBoxLayout()
        
        # Add description
        description = QLabel("Clean and standardize the processed training data.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add controls
        controls_group = QGroupBox("Controls")
        controls_layout = QFormLayout()
        
        self.clean_input_file = QComboBox()
        self.clean_input_file.addItems(["trainingdata2.txt", "trainingdata_partial2.txt"])
        self.clean_input_file.setEditable(True)
        controls_layout.addRow("Input File:", self.clean_input_file)
        
        self.clean_output_file = QComboBox()
        self.clean_output_file.addItems(["cleaned_trainingdata.txt"])
        self.clean_output_file.setEditable(True)
        controls_layout.addRow("Output File:", self.clean_output_file)
        
        button_layout = QHBoxLayout()
        self.browse_input_button = QPushButton("Browse Input")
        self.browse_input_button.clicked.connect(self.browse_input_file)
        button_layout.addWidget(self.browse_input_button)
        
        self.browse_output_button = QPushButton("Browse Output")
        self.browse_output_button.clicked.connect(self.browse_output_file)
        button_layout.addWidget(self.browse_output_button)
        
        self.clean_button = QPushButton("Start Cleaning")
        self.clean_button.clicked.connect(self.start_cleaning)
        button_layout.addWidget(self.clean_button)
        
        controls_layout.addRow("", button_layout)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Add output area
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        self.clean_output = QTextEdit()
        self.clean_output.setReadOnly(True)
        output_layout.addWidget(self.clean_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        clean_tab.setLayout(layout)
        self.tabs.addTab(clean_tab, "Clean Data")
    
    def create_train_tab(self):
        # Create the Train Model tab
        train_tab = QWidget()
        layout = QVBoxLayout()
        
        # Add description
        description = QLabel("Train the JAImes Madison model on the cleaned data.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add controls
        controls_group = QGroupBox("Training Parameters")
        controls_layout = QFormLayout()
        
        self.model_name = QComboBox()
        self.model_name.addItems(["gpt2", "gpt2-medium", "gpt2-large"])
        self.model_name.setCurrentText("gpt2-medium")
        controls_layout.addRow("Model:", self.model_name)
        
        self.num_epochs = QSpinBox()
        self.num_epochs.setRange(1, 20)
        self.num_epochs.setValue(8)
        controls_layout.addRow("Epochs:", self.num_epochs)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(4)
        controls_layout.addRow("Batch Size:", self.batch_size)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(1e-7, 1e-4)
        self.learning_rate.setValue(5e-6)
        self.learning_rate.setDecimals(7)
        self.learning_rate.setSingleStep(1e-6)
        controls_layout.addRow("Learning Rate:", self.learning_rate)
        
        self.display_qa_pairs = QCheckBox()
        self.display_qa_pairs.setChecked(False)
        controls_layout.addRow("Display QA Pairs:", self.display_qa_pairs)
        
        self.use_high_quality = QCheckBox()
        self.use_high_quality.setChecked(True)
        controls_layout.addRow("Use High Quality Dataset:", self.use_high_quality)
        
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addRow("", self.train_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Add output area
        output_group = QGroupBox("Training Output")
        output_layout = QVBoxLayout()
        
        self.train_output = QTextEdit()
        self.train_output.setReadOnly(True)
        output_layout.addWidget(self.train_output)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        train_tab.setLayout(layout)
        self.tabs.addTab(train_tab, "Train Model")
    
    def create_chat_tab(self):
        # Create the Chat tab
        chat_tab = QWidget()
        layout = QVBoxLayout()
        
        if AIModelChat is not None:
            # Create an instance of the chat interface
            self.chat_widget = AIModelChat()
            layout.addWidget(self.chat_widget)
        else:
            # Display a message if the chat module is not available
            message = QLabel("Chat functionality is not available. Make sure chat_gui.py is in the same directory.")
            message.setAlignment(Qt.AlignCenter)
            layout.addWidget(message)
            
            # Add a button to launch the chat in a separate process
            launch_button = QPushButton("Launch Chat in Separate Window")
            launch_button.clicked.connect(self.launch_chat_external)
            layout.addWidget(launch_button)
        
        chat_tab.setLayout(layout)
        self.tabs.addTab(chat_tab, "Chat")
    
    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Text Files (*.txt)")
        if file_path:
            self.clean_input_file.setCurrentText(file_path)
    
    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "Text Files (*.txt)")
        if file_path:
            self.clean_output_file.setCurrentText(file_path)
    
    def start_processing(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Process Running", "A process is already running. Please wait for it to complete.")
            return
        
        self.process_output.clear()
        self.process_button.setEnabled(False)
        
        # Create and start the worker thread
        self.worker = ProcessWorker("process_data")
        self.worker.update_signal.connect(self.update_process_output)
        self.worker.finished_signal.connect(self.process_finished)
        self.worker.start()
    
    def start_cleaning(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Process Running", "A process is already running. Please wait for it to complete.")
            return
        
        self.clean_output.clear()
        self.clean_button.setEnabled(False)
        
        # Get parameters
        params = {
            'input_file': self.clean_input_file.currentText(),
            'output_file': self.clean_output_file.currentText()
        }
        
        # Create and start the worker thread
        self.worker = ProcessWorker("clean_data", params)
        self.worker.update_signal.connect(self.update_clean_output)
        self.worker.finished_signal.connect(self.clean_finished)
        self.worker.start()
    
    def start_training(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Process Running", "A process is already running. Please wait for it to complete.")
            return
        
        self.train_output.clear()
        self.train_button.setEnabled(False)
        
        # Get parameters
        params = {
            'model_name': self.model_name.currentText(),
            'num_epochs': self.num_epochs.value(),
            'batch_size': self.batch_size.value(),
            'learning_rate': self.learning_rate.value(),
            'display_qa_pairs': self.display_qa_pairs.isChecked(),
            'use_high_quality': self.use_high_quality.isChecked()
        }
        
        # Create and start the worker thread
        self.worker = ProcessWorker("train_model", params)
        self.worker.update_signal.connect(self.update_train_output)
        self.worker.finished_signal.connect(self.train_finished)
        self.worker.start()
    
    def launch_chat_external(self):
        try:
            # Try to find the chat_gui.py file
            if os.path.exists("chat_gui.py"):
                subprocess.Popen([sys.executable, "chat_gui.py"])
            else:
                QMessageBox.warning(self, "File Not Found", "chat_gui.py not found in the current directory.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch chat: {str(e)}")
    
    @pyqtSlot(str)
    def update_process_output(self, text):
        self.process_output.append(text)
    
    @pyqtSlot(str)
    def update_clean_output(self, text):
        self.clean_output.append(text)
    
    @pyqtSlot(str)
    def update_train_output(self, text):
        self.train_output.append(text)
    
    @pyqtSlot(bool, str)
    def process_finished(self, success, message):
        self.process_button.setEnabled(True)
        if success:
            self.process_output.append("\n✅ " + message)
        else:
            self.process_output.append("\n❌ " + message)
    
    @pyqtSlot(bool, str)
    def clean_finished(self, success, message):
        self.clean_button.setEnabled(True)
        if success:
            self.clean_output.append("\n✅ " + message)
        else:
            self.clean_output.append("\n❌ " + message)
    
    @pyqtSlot(bool, str)
    def train_finished(self, success, message):
        self.train_button.setEnabled(True)
        if success:
            self.train_output.append("\n✅ " + message)
        else:
            self.train_output.append("\n❌ " + message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterGUI()
    window.show()
    sys.exit(app.exec_()) 