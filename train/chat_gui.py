#!/usr/bin/env python3
"""
Chat GUI for JAImes Madison AI - Provides an interface for chatting with the trained model
"""

import sys
import os
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                            QComboBox, QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QIcon

# Try to import the transformers library
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ModelThread(QThread):
    """Thread for generating responses from the model without blocking the UI"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, tokenizer, prompt, max_length=200, temperature=0.7):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.temperature = temperature
    
    def run(self):
        try:
            # Prepare the prompt
            inputs = self.tokenizer.encode(self.prompt, return_tensors="pt")
            
            # Move to the same device as the model
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate response
            attention_mask = torch.ones(inputs.shape, device=device)
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=self.max_length + inputs.shape[1],
                temperature=self.temperature,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Emit the response
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))

class AIModelChat(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JAImes Madison AI Chat")
        self.resize(800, 600)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_thread = None
        
        # Set up the UI
        self.setup_ui()
        
        # Try to load the model if available
        if TRANSFORMERS_AVAILABLE:
            self.try_load_default_model()
    
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Model selection area
        model_layout = QHBoxLayout()
        
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)
        
        self.model_path = QComboBox()
        self.model_path.setEditable(True)
        
        # Add default paths
        default_paths = ["madison_model"]
        if os.path.exists("madison_model"):
            default_paths.insert(0, "madison_model")
        
        self.model_path.addItems(default_paths)
        model_layout.addWidget(self.model_path, 1)
        
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_button)
        
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_button)
        
        main_layout.addLayout(model_layout)
        
        # Chat area
        chat_splitter = QSplitter(Qt.Vertical)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: Arial, sans-serif;
            }
        """)
        chat_splitter.addWidget(self.chat_history)
        
        # Input area
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # User input
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.setMinimumHeight(100)
        self.user_input.setMaximumHeight(150)
        self.user_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: Arial, sans-serif;
            }
        """)
        input_layout.addWidget(self.user_input)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear Chat")
        self.clear_button.clicked.connect(self.clear_chat)
        controls_layout.addWidget(self.clear_button)
        
        controls_layout.addStretch()
        
        self.temperature_label = QLabel("Temperature:")
        controls_layout.addWidget(self.temperature_label)
        
        self.temperature = QComboBox()
        self.temperature.addItems(["0.5", "0.7", "0.9", "1.0", "1.2"])
        self.temperature.setCurrentText("0.7")
        controls_layout.addWidget(self.temperature)
        
        self.max_length_label = QLabel("Max Length:")
        controls_layout.addWidget(self.max_length_label)
        
        self.max_length = QComboBox()
        self.max_length.addItems(["100", "200", "300", "500", "1000"])
        self.max_length.setCurrentText("200")
        controls_layout.addWidget(self.max_length)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)  # Disabled until model is loaded
        controls_layout.addWidget(self.send_button)
        
        input_layout.addLayout(controls_layout)
        chat_splitter.addWidget(input_widget)
        
        # Set initial sizes
        chat_splitter.setSizes([400, 200])
        
        main_layout.addWidget(chat_splitter)
        
        # Status bar
        self.status_bar = QLabel("Model not loaded")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border-top: 1px solid #ddd;
                padding: 5px;
            }
        """)
        main_layout.addWidget(self.status_bar)
        
        self.setLayout(main_layout)
        
        # Connect enter key to send message
        self.user_input.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        if obj is self.user_input and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)
    
    def try_load_default_model(self):
        """Try to load the default model if it exists"""
        if os.path.exists("madison_model") and os.path.exists("madison_tokenizer"):
            self.load_model_from_path("madison_model")
        elif os.path.exists("madison_model"):
            self.load_model_from_path("madison_model")
    
    def browse_model(self):
        """Open a file dialog to select a model directory"""
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if model_dir:
            self.model_path.setCurrentText(model_dir)
    
    def load_model(self):
        """Load the model from the specified path"""
        model_path = self.model_path.currentText()
        self.load_model_from_path(model_path)
    
    def load_model_from_path(self, model_path):
        """Load the model and tokenizer from the specified path"""
        if not TRANSFORMERS_AVAILABLE:
            QMessageBox.critical(self, "Error", "The transformers library is not installed. Please install it with pip install transformers.")
            return
        
        try:
            self.status_bar.setText("Loading model...")
            QApplication.processEvents()
            
            # Check if the model path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist")
            
            # Load the model and tokenizer
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            
            # Try to load the tokenizer from the same directory or a tokenizer directory
            tokenizer_path = model_path
            if os.path.exists(os.path.join(model_path, "tokenizer")):
                tokenizer_path = os.path.join(model_path, "tokenizer")
            elif os.path.exists("madison_tokenizer"):
                tokenizer_path = "madison_tokenizer"
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Enable the send button
            self.send_button.setEnabled(True)
            
            # Update status
            self.status_bar.setText(f"Model loaded from {model_path} (Device: {device})")
            
            # Add a welcome message
            self.chat_history.append("<b>JAImes Madison:</b> Greetings! I am JAImes Madison, primary architect of the U.S. Constitution and fourth President of the United States. How may I assist you today?")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_bar.setText(f"Error loading model: {str(e)}")
    
    def send_message(self):
        """Send the user's message to the model and get a response"""
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a model first.")
            return
        
        # Get the user's message
        user_message = self.user_input.toPlainText().strip()
        if not user_message:
            return
        
        # Disable the send button while generating
        self.send_button.setEnabled(False)
        self.status_bar.setText("Generating response...")
        
        # Display the user's message
        self.chat_history.append(f"<b>You:</b> {user_message}")
        
        # Clear the input field
        self.user_input.clear()
        
        # Prepare the prompt
        # Format: Previous conversation + new user message
        conversation_history = self.get_conversation_history()
        prompt = f"{conversation_history}You: {user_message}\n\nJAImes Madison:"
        
        # Get parameters
        temperature = float(self.temperature.currentText())
        max_length = int(self.max_length.currentText())
        
        # Generate response in a separate thread
        self.model_thread = ModelThread(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_length=max_length, 
            temperature=temperature
        )
        self.model_thread.response_ready.connect(self.handle_response)
        self.model_thread.error_occurred.connect(self.handle_error)
        self.model_thread.start()
    
    def get_conversation_history(self):
        """Extract the conversation history from the chat history widget"""
        # This is a simple implementation that works for short conversations
        # For longer conversations, you might need to implement a more sophisticated approach
        # that keeps track of the conversation history separately
        history_text = self.chat_history.toPlainText()
        
        # Limit the history to the last 5 exchanges to avoid context length issues
        lines = history_text.split('\n')
        if len(lines) > 10:  # Roughly 5 exchanges (user + model)
            lines = lines[-10:]
            history_text = '\n'.join(lines)
        
        # Format the history
        history = history_text.replace("You:", "You:").replace("JAImes Madison:", "JAImes Madison:")
        
        if history:
            return history + "\n\n"
        return ""
    
    def handle_response(self, response):
        """Handle the response from the model"""
        # Clean up the response
        response = response.strip()
        
        # Display the response
        self.chat_history.append(f"<b>JAImes Madison:</b> {response}")
        
        # Scroll to the bottom
        self.chat_history.moveCursor(QTextCursor.End)
        
        # Re-enable the send button
        self.send_button.setEnabled(True)
        self.status_bar.setText("Ready")
    
    def handle_error(self, error_message):
        """Handle errors during response generation"""
        QMessageBox.critical(self, "Error", f"Error generating response: {error_message}")
        self.chat_history.append("<i>Error generating response. Please try again.</i>")
        self.send_button.setEnabled(True)
        self.status_bar.setText(f"Error: {error_message}")
    
    def clear_chat(self):
        """Clear the chat history"""
        self.chat_history.clear()
        self.chat_history.append("<b>JAImes Madison:</b> Greetings! I am JAImes Madison, primary architect of the U.S. Constitution and fourth President of the United States. How may I assist you today?")

class ChatMainWindow(QMainWindow):
    """Main window for the chat application when run standalone"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JAImes Madison AI Chat")
        self.resize(800, 600)
        
        # Create the chat widget
        self.chat_widget = AIModelChat()
        self.setCentralWidget(self.chat_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatMainWindow()
    window.show()
    sys.exit(app.exec_()) 