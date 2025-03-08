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
import glob
import threading
import time
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
                            QProgressBar, QGroupBox, QFormLayout, QListWidget, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QPixmap

# Import the modules we need
try:
    from chat_gui import AIModelChat, get_model_activation_data
except ImportError:
    print("Warning: chat_gui module not found. Chat functionality will be disabled.")
    AIModelChat = None
    get_model_activation_data = None

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
            from processdata import test_api_connection, chunk_text, process_chunk, process_file
            
            # Get input and output file paths from parameters
            input_file = self.params.get('input_file', 'federalist_papers.txt')
            output_file = self.params.get('output_file', 'trainingdata2.txt')
            
            # Get chunking parameters
            auto_chunk = self.params.get('auto_chunk', True)
            max_tokens = self.params.get('max_tokens', 2000)
            chunk_size = self.params.get('chunk_size', None)
            overlap = self.params.get('overlap', 0.2)
            
            # Redirect stdout to our signal
            original_stdout = sys.stdout
            sys.stdout = self
            
            self.update_signal.emit(f"Processing file: {input_file}")
            self.update_signal.emit(f"Output will be saved to: {output_file}")
            
            if auto_chunk:
                self.update_signal.emit(f"Using automatic chunking with max_tokens={max_tokens}")
            else:
                self.update_signal.emit(f"Using custom chunk size: {chunk_size} characters with {overlap*100}% overlap")
            
            # Check if the process_file function exists in the module
            if hasattr(process_file, '__call__'):
                # Use the process_file function if it exists
                process_file(
                    input_file=input_file, 
                    output_file=output_file,
                    max_tokens=max_tokens,
                    auto_chunk=auto_chunk,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
            else:
                # Otherwise, try to use the main function
                from processdata import main as m
                # Set environment variables or globals if needed to specify the files
                os.environ['INPUT_FILE'] = input_file
                os.environ['OUTPUT_FILE'] = output_file
                m()
            
            sys.stdout = original_stdout
            
        except ImportError as e:
            self.update_signal.emit(f"Error: processdata.py module not found. {e}")
        except Exception as e:
            self.update_signal.emit(f"Error processing data: {str(e)}")
    
    def run_clean_data(self):
        # Import and run the cleandata module
        try:
            from cleandata import clean_qa_pairs
            # Redirect stdout to our signal
            original_stdout = sys.stdout
            sys.stdout = self
            
            # Get parameters
            input_file = self.params.get('input_file', 'trainingdata2.txt')
            output_file = self.params.get('output_file', 'cleaned_trainingdata.txt')
            
            # Call the clean_qa_pairs function with basic parameters
            clean_qa_pairs(
                input_file=input_file,
                output_file=output_file
            )
            
            sys.stdout = original_stdout
        except ImportError as e:
            self.update_signal.emit(f"Error: cleandata.py module not found. {e}")
        except Exception as e:
            self.update_signal.emit(f"Error cleaning data: {str(e)}")
    
    def run_train_model(self):
        # Import and run the train module
        try:
            from train import train_model
            # Redirect stdout to our signal
            original_stdout = sys.stdout
            sys.stdout = self
            
            # Get parameters
            input_file = self.params.get('input_file', 'cleaned_trainingdata.txt')
            model_name = self.params.get('model_name', 'gpt2-medium')
            num_epochs = self.params.get('num_epochs', 5)
            batch_size = self.params.get('batch_size', 4)
            learning_rate = self.params.get('learning_rate', 5e-6)
            display_qa_pairs = self.params.get('display_qa_pairs', False)
            use_high_quality = self.params.get('use_high_quality', True)
            
            self.update_signal.emit(f"Training with file: {input_file}")
            self.update_signal.emit(f"Model: {model_name}, Epochs: {num_epochs}, Batch Size: {batch_size}")
            
            # Check if the train_model function accepts an input_file parameter
            import inspect
            train_params = inspect.signature(train_model).parameters
            
            if 'input_file' in train_params:
                # If the function accepts input_file, pass it
                train_model(
                    input_file=input_file,
                    model_name=model_name,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    display_qa_pairs=display_qa_pairs,
                    use_high_quality=use_high_quality
                )
            else:
                # Otherwise, set an environment variable and call without input_file
                self.update_signal.emit(f"Note: Using environment variable for input file")
                os.environ['TRAINING_FILE'] = input_file
                
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
        except Exception as e:
            self.update_signal.emit(f"Error training model: {str(e)}")
    
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
        
        # Allow the window to be resized to a smaller size
        self.setMinimumSize(600, 400)  # Set a reasonable minimum size
        
        # Create the main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Initialize activation data storage
        self.activation_data = {}
        self.activation_files = []
        
        # Create tabs for each functionality
        self.create_process_tab()
        self.create_clean_tab()
        self.create_train_tab()
        self.create_chat_tab()
        self.create_visualization_tab()
        
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
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QLabel {
                color: #333333;
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
        
        # Add input file selection
        input_group = QGroupBox("Input File")
        input_layout = QHBoxLayout()
        
        self.process_input_file = QComboBox()
        # Add default files
        default_files = ["federalist_papers.txt", "federalist_papers_full.txt"]
        self.process_input_file.addItems(default_files)
        self.process_input_file.setEditable(True)
        input_layout.addWidget(self.process_input_file, 1)
        
        self.browse_process_input_button = QPushButton("Browse")
        self.browse_process_input_button.clicked.connect(self.browse_process_input_file)
        input_layout.addWidget(self.browse_process_input_button)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Add output file selection
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout()
        
        self.process_output_file = QComboBox()
        # Add default output files
        default_output_files = ["trainingdata2.txt", "trainingdata_new.txt"]
        self.process_output_file.addItems(default_output_files)
        self.process_output_file.setEditable(True)
        output_layout.addWidget(self.process_output_file, 1)
        
        self.browse_process_output_button = QPushButton("Browse")
        self.browse_process_output_button.clicked.connect(self.browse_process_output_file)
        output_layout.addWidget(self.browse_process_output_button)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Add chunking options
        chunking_group = QGroupBox("Chunking Options")
        chunking_layout = QFormLayout()
        
        # Auto chunk checkbox
        self.process_auto_chunk_checkbox = QCheckBox("Auto Chunk")
        self.process_auto_chunk_checkbox.setChecked(True)
        self.process_auto_chunk_checkbox.stateChanged.connect(self.toggle_process_chunk_size)
        chunking_layout.addRow("Auto Chunking:", self.process_auto_chunk_checkbox)
        
        # Max tokens for auto chunking
        self.process_max_tokens_spinbox = QSpinBox()
        self.process_max_tokens_spinbox.setRange(500, 8000)
        self.process_max_tokens_spinbox.setValue(2000)
        self.process_max_tokens_spinbox.setSingleStep(100)
        chunking_layout.addRow("Max Tokens:", self.process_max_tokens_spinbox)
        
        # Custom chunk size
        self.process_chunk_size_spinbox = QSpinBox()
        self.process_chunk_size_spinbox.setRange(1000, 50000)
        self.process_chunk_size_spinbox.setValue(8000)
        self.process_chunk_size_spinbox.setSingleStep(1000)
        self.process_chunk_size_spinbox.setEnabled(False)  # Disabled when auto-chunk is on
        chunking_layout.addRow("Chunk Size (chars):", self.process_chunk_size_spinbox)
        
        # Overlap
        self.process_overlap_spinbox = QDoubleSpinBox()
        self.process_overlap_spinbox.setRange(0.0, 0.9)
        self.process_overlap_spinbox.setValue(0.2)
        self.process_overlap_spinbox.setSingleStep(0.1)
        chunking_layout.addRow("Overlap:", self.process_overlap_spinbox)
        
        chunking_group.setLayout(chunking_layout)
        layout.addWidget(chunking_group)
        
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
        
        # Add input file selection
        input_group = QGroupBox("Training Data")
        input_layout = QHBoxLayout()
        
        self.train_input_file = QComboBox()
        # Add default files
        default_files = ["cleaned_trainingdata.txt"]
        # Add any other training files in the directory
        training_files = [f for f in self.list_files(".", ["*.txt"]) if "training" in f.lower() or "cleaned" in f.lower()]
        for file in training_files:
            if file not in default_files:
                default_files.append(file)
        self.train_input_file.addItems(default_files)
        self.train_input_file.setEditable(True)
        input_layout.addWidget(self.train_input_file, 1)
        
        self.browse_train_input_button = QPushButton("Browse")
        self.browse_train_input_button.clicked.connect(self.browse_train_input_file)
        input_layout.addWidget(self.browse_train_input_button)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
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
            # Create a splitter to allow resizing between chat and controls
            splitter = QSplitter(Qt.Vertical)
            
            # Create an instance of the chat interface
            self.chat_widget = AIModelChat()
            # Remove any minimum height constraints
            self.chat_widget.setMinimumHeight(100)  # Set a small minimum height
            splitter.addWidget(self.chat_widget)
            
            # Add controls for capturing activations in a more compact layout
            activation_widget = QWidget()
            activation_layout = QHBoxLayout(activation_widget)
            activation_layout.setContentsMargins(5, 5, 5, 5)
            
            activation_label = QLabel("Model Visualization:")
            activation_layout.addWidget(activation_label)
            
            self.capture_activations_button = QPushButton("Capture Current Activations")
            self.capture_activations_button.clicked.connect(self.capture_activations)
            activation_layout.addWidget(self.capture_activations_button)
            
            # Add stretch to push controls to the left
            activation_layout.addStretch()
            
            # Set a small fixed height for the activation controls
            activation_widget.setFixedHeight(40)
            splitter.addWidget(activation_widget)
            
            # Set initial sizes to give most space to the chat widget
            splitter.setSizes([600, 40])
            
            layout.addWidget(splitter)
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
    
    def create_visualization_tab(self):
        # Create the Visualization tab
        visualization_tab = QWidget()
        layout = QVBoxLayout()
        
        # Create a main splitter to allow resizing of all sections
        main_splitter = QSplitter(Qt.Vertical)
        
        # Add description
        description = QLabel("Visualize model activations and other data.")
        description.setWordWrap(True)
        description.setMaximumHeight(30)
        layout.addWidget(description)
        
        # Top section: Data selection
        data_selection_widget = QWidget()
        data_layout = QVBoxLayout(data_selection_widget)
        data_layout.setContentsMargins(5, 5, 5, 5)
        data_layout.setSpacing(5)
        
        # Add a label for activation files
        activation_label = QLabel("Model Activation Files:")
        data_layout.addWidget(activation_label)
        
        # Add a list widget for activation files
        self.viz_file_list = QListWidget()
        self.viz_file_list.setMinimumHeight(50)  # Allow to be resized smaller
        self.viz_file_list.itemSelectionChanged.connect(self.activation_file_selected)
        data_layout.addWidget(self.viz_file_list)
        
        # Add other data files
        other_data_layout = QFormLayout()
        other_data_layout.setSpacing(5)
        self.viz_data_file = QComboBox()
        data_files = self.list_files(".", ["*.txt", "*.csv"])
        self.viz_data_file.addItems(data_files)
        self.viz_data_file.setEditable(True)
        other_data_layout.addRow("Other Data File:", self.viz_data_file)
        
        self.browse_viz_file_button = QPushButton("Browse")
        self.browse_viz_file_button.clicked.connect(self.browse_viz_file)
        other_data_layout.addRow("", self.browse_viz_file_button)
        
        data_layout.addLayout(other_data_layout)
        main_splitter.addWidget(data_selection_widget)
        
        # Middle section: Visualization controls
        viz_controls_widget = QWidget()
        viz_controls_layout = QFormLayout(viz_controls_widget)
        viz_controls_layout.setSpacing(5)
        
        self.chart_type = QComboBox()
        self.chart_type.addItems([
            "Attention Heatmap", 
            "Layer Activations", 
            "Neuron Activations", 
            "Attention Heads", 
            "Token Embeddings",
            "Bar Chart", 
            "Line Chart", 
            "Pie Chart", 
            "Scatter Plot", 
            "Heatmap"
        ])
        viz_controls_layout.addRow("Chart Type:", self.chart_type)
        
        self.layer_selector = QComboBox()
        self.layer_selector.addItems(["All Layers", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5"])
        viz_controls_layout.addRow("Layer:", self.layer_selector)
        
        self.head_selector = QComboBox()
        self.head_selector.addItems(["All Heads", "Head 1", "Head 2", "Head 3", "Head 4"])
        viz_controls_layout.addRow("Attention Head:", self.head_selector)
        
        self.x_axis = QComboBox()
        self.x_axis.addItems(["Neurons", "Tokens", "Layers", "Heads"])
        viz_controls_layout.addRow("X-Axis:", self.x_axis)
        
        self.y_axis = QComboBox()
        self.y_axis.addItems(["Activation Value", "Attention Score", "Gradient", "Magnitude"])
        viz_controls_layout.addRow("Y-Axis:", self.y_axis)
        
        self.generate_viz_button = QPushButton("Generate Visualization")
        self.generate_viz_button.clicked.connect(self.generate_visualization)
        viz_controls_layout.addRow("", self.generate_viz_button)
        
        main_splitter.addWidget(viz_controls_widget)
        
        # Bottom section: Visualization display
        viz_display_widget = QWidget()
        viz_display_layout = QVBoxLayout(viz_display_widget)
        viz_display_layout.setContentsMargins(5, 5, 5, 5)
        
        self.viz_display = QWidget()
        self.viz_display.setMinimumHeight(100)  # Allow to be resized smaller
        self.viz_display.setStyleSheet("background-color: white;")
        
        # Create a layout for the visualization display
        self.viz_display_layout = QVBoxLayout(self.viz_display)
        
        # Create a matplotlib figure and canvas
        self.figure = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.viz_display_layout.addWidget(self.canvas)
        
        viz_display_layout.addWidget(self.viz_display)
        
        # Add export button
        export_layout = QHBoxLayout()
        self.export_viz_button = QPushButton("Export as PNG")
        self.export_viz_button.clicked.connect(self.export_visualization)
        export_layout.addWidget(self.export_viz_button)
        export_layout.addStretch()
        viz_display_layout.addLayout(export_layout)
        
        main_splitter.addWidget(viz_display_widget)
        
        # Set initial sizes for the splitter
        main_splitter.setSizes([150, 150, 300])
        
        layout.addWidget(main_splitter)
        visualization_tab.setLayout(layout)
        self.tabs.addTab(visualization_tab, "Visualizations")
        
        # Update the file list
        self.update_viz_file_list()
    
    def update_viz_file_list(self):
        """Update the list of activation files in the visualization tab"""
        self.viz_file_list.clear()
        for filename in self.activation_files:
            self.viz_file_list.addItem(filename)
    
    def activation_file_selected(self):
        """Handle selection of an activation file"""
        selected_items = self.viz_file_list.selectedItems()
        if not selected_items:
            return
            
        filename = selected_items[0].text()
        if filename in self.activation_data:
            # Update the UI based on the selected file
            data = self.activation_data[filename]
            QMessageBox.information(self, "File Info", 
                                   f"Selected: {filename}\nQuery: {data['query']}\nTimestamp: {data['timestamp']}")
    
    def list_files(self, directory, extensions):
        """List files in the given directory with the specified extensions."""
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        return files

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
        
        # Get the input and output file paths
        input_file = self.process_input_file.currentText()
        output_file = self.process_output_file.currentText()
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            QMessageBox.critical(self, "Error", f"Input file '{input_file}' does not exist.")
            return
        
        self.process_output.clear()
        self.process_button.setEnabled(False)
        
        # Get chunking parameters
        auto_chunk = self.process_auto_chunk_checkbox.isChecked()
        max_tokens = self.process_max_tokens_spinbox.value()
        chunk_size = self.process_chunk_size_spinbox.value() if not auto_chunk else None
        overlap = self.process_overlap_spinbox.value()
        
        # Create and start the worker thread with all parameters
        params = {
            'input_file': input_file,
            'output_file': output_file,
            'auto_chunk': auto_chunk,
            'max_tokens': max_tokens,
            'chunk_size': chunk_size,
            'overlap': overlap
        }
        self.worker = ProcessWorker("process_data", params)
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
        
        # Get the input file path
        input_file = self.train_input_file.currentText()
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            QMessageBox.critical(self, "Error", f"Input file '{input_file}' does not exist.")
            return
        
        self.train_output.clear()
        self.train_button.setEnabled(False)
        
        # Get parameters
        params = {
            'input_file': input_file,
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
    
    def browse_viz_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.viz_data_file.setCurrentText(file_path)
    
    def generate_visualization(self):
        """Generate visualization based on the selected options"""
        # Clear the current figure
        self.figure.clear()
        
        # Get the selected visualization type
        chart_type = self.chart_type.currentText()
        
        # Check if we're visualizing activation data
        selected_items = self.viz_file_list.selectedItems()
        if selected_items:
            filename = selected_items[0].text()
            if filename in self.activation_data:
                self.generate_activation_visualization(filename, chart_type)
                return
        
        # Otherwise, use the other data file
        data_file = self.viz_data_file.currentText()
        x_axis = self.x_axis.currentText()
        y_axis = self.y_axis.currentText()
        
        # For demonstration, create a simple plot
        ax = self.figure.add_subplot(111)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title(f"{chart_type}: {data_file}")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        
        # Refresh the canvas
        self.canvas.draw()
    
    def generate_activation_visualization(self, filename, chart_type):
        """Generate visualization for activation data"""
        data = self.activation_data[filename]['data']
        query = self.activation_data[filename]['query']
        
        # Get selected layer and head
        layer_text = self.layer_selector.currentText()
        head_text = self.head_selector.currentText()
        
        layer = None if layer_text == "All Layers" else int(layer_text.split()[-1]) - 1
        head = None if head_text == "All Heads" else int(head_text.split()[-1]) - 1
        
        # Create the appropriate visualization based on chart type
        ax = self.figure.add_subplot(111)
        
        if chart_type == "Attention Heatmap":
            # Example: create a heatmap of attention weights
            if 'attention_weights' in data:
                attention = data['attention_weights']
                if layer is not None and head is not None and len(attention.shape) >= 4:
                    # Extract specific layer and head
                    attn_map = attention[layer, head]
                else:
                    # Average across layers and heads
                    attn_map = np.mean(attention, axis=(0, 1))
                
                im = ax.imshow(attn_map, cmap='viridis')
                ax.set_title(f"Attention Weights for: {query}")
                self.figure.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, "No attention data available", 
                       horizontalalignment='center', verticalalignment='center')
        
        elif chart_type == "Layer Activations":
            # Example: plot average activation per layer
            if 'hidden_states' in data:
                hidden = data['hidden_states']
                if len(hidden.shape) >= 3:
                    # Average across tokens and features
                    layer_activations = np.mean(np.abs(hidden), axis=(1, 2))
                    ax.bar(range(len(layer_activations)), layer_activations)
                    ax.set_title(f"Average Layer Activation for: {query}")
                    ax.set_xlabel("Layer")
                    ax.set_ylabel("Average Activation")
            else:
                ax.text(0.5, 0.5, "No hidden state data available", 
                       horizontalalignment='center', verticalalignment='center')
        
        elif chart_type == "Neuron Activations":
            # Example: plot neuron activations for a specific layer
            if 'hidden_states' in data and layer is not None:
                hidden = data['hidden_states']
                if len(hidden.shape) >= 3 and layer < len(hidden):
                    # Average across tokens
                    neuron_activations = np.mean(hidden[layer], axis=0)
                    ax.plot(neuron_activations)
                    ax.set_title(f"Neuron Activations for Layer {layer+1}")
                    ax.set_xlabel("Neuron Index")
                    ax.set_ylabel("Activation")
            else:
                ax.text(0.5, 0.5, "No hidden state data available or invalid layer", 
                       horizontalalignment='center', verticalalignment='center')
        
        else:
            # Default visualization
            ax.text(0.5, 0.5, f"Visualization type '{chart_type}' not implemented", 
                   horizontalalignment='center', verticalalignment='center')
        
        # Refresh the canvas
        self.canvas.draw()
    
    def export_visualization(self):
        """Export the current visualization to a file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Visualization", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Visualization saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save visualization: {str(e)}")
    
    def capture_activations(self):
        """Capture the current model activations from the chat interface"""
        if not hasattr(self, 'chat_widget') or get_model_activation_data is None:
            QMessageBox.warning(self, "Not Available", "Activation capture is not available.")
            return
            
        try:
            # Get the current query from the chat widget
            current_query = self.chat_widget.get_current_query()
            if not current_query:
                QMessageBox.warning(self, "No Query", "Please enter a query in the chat interface first.")
                return
                
            # Get activation data from the model
            activation_data = get_model_activation_data(current_query)
            
            # Save the activation data
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"activation_data_{timestamp}.npz"
            np.savez(filename, **activation_data)
            
            # Add to our list of activation files
            self.activation_files.append(filename)
            self.activation_data[filename] = {
                'query': current_query,
                'timestamp': timestamp,
                'data': activation_data
            }
            
            # Update the visualization tab's file list
            self.update_viz_file_list()
            
            QMessageBox.information(self, "Success", f"Activation data saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to capture activations: {str(e)}")

    def browse_process_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.process_input_file.setCurrentText(file_path)
    
    def browse_process_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.process_output_file.setCurrentText(file_path)

    def toggle_process_chunk_size(self, state):
        """Enable or disable the process chunk size spinbox based on auto-chunk checkbox state"""
        self.process_chunk_size_spinbox.setEnabled(not state)
        self.process_max_tokens_spinbox.setEnabled(state)

    def browse_train_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Training Data File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.train_input_file.setCurrentText(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterGUI()
    window.show()
    sys.exit(app.exec_()) 