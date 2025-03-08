import sys
import torch
import numpy as np
import time
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                            QTabWidget, QLabel, QComboBox, QSplitter, QFileDialog,
                            QMessageBox, QProgressBar, QStatusBar, QAction, QMenu, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ModelThread(QThread):
    """Thread for running model inference without blocking the UI"""
    response_signal = pyqtSignal(str)
    token_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    
    def __init__(self, model, tokenizer, prompt, max_length=200, temperature=0.7):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run(self):
        try:
            # Encode the prompt
            inputs = self.tokenizer(self.prompt, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # First generate the full response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length + len(input_ids[0]),
                    temperature=self.temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    min_length=len(input_ids[0]) + 10,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                )
            
            # Get the full generated text
            full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the response part (after the prompt)
            response = full_text[len(self.prompt):].strip()
            
            # Emit the full response for saving
            self.response_signal.emit(response)
            
            # Simulate streaming by emitting character by character
            for i, char in enumerate(response):
                self.token_signal.emit(char)
                # Update progress
                progress = int((i / len(response)) * 100)
                self.progress_signal.emit(progress)
                # Small delay to make the streaming more visible
                time.sleep(0.01)
            
            self.progress_signal.emit(100)
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))


class ActivationVisualizer(QWidget):
    """Widget for visualizing model activations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Layer selection
        layer_layout = QHBoxLayout()
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("All Layers")
        layer_layout.addWidget(QLabel("Layer:"))
        layer_layout.addWidget(self.layer_combo)
        
        # Visualization buttons
        viz_layout = QHBoxLayout()
        self.visualize_btn = QPushButton("Visualize Current")
        self.compare_btn = QPushButton("Compare with Saved")
        self.save_btn = QPushButton("Save Current")
        viz_layout.addWidget(self.visualize_btn)
        viz_layout.addWidget(self.compare_btn)
        viz_layout.addWidget(self.save_btn)
        
        # Figure for matplotlib
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add all to main layout
        layout.addLayout(layer_layout)
        layout.addLayout(viz_layout)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)


class AIModelChat(QMainWindow):
    """Main application window for the AI chat interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat Interface")
        self.resize(1000, 800)
        
        # Model attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.activations = {}
        self.hooks = []
        self.activation_history = {}
        self.message_history = []
        
        # Setup UI
        self.setup_ui()
        
        # Load model
        self.load_model()
        
    def setup_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Chat tab
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 10))
        
        # Input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        
        # Progress bar for generation
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Add widgets to chat layout
        chat_layout.addWidget(self.chat_history)
        chat_layout.addLayout(input_layout)
        chat_layout.addWidget(self.progress_bar)
        
        # Visualization tab
        self.viz_widget = ActivationVisualizer()
        
        # Add tabs
        self.tabs.addTab(chat_widget, "Chat")
        self.tabs.addTab(self.viz_widget, "Visualizations")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Menu bar
        self.setup_menu()
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Connect signals
        self.viz_widget.visualize_btn.clicked.connect(self.visualize_activations)
        self.viz_widget.compare_btn.clicked.connect(self.compare_activations)
        self.viz_widget.save_btn.clicked.connect(self.save_activations)
    
    def setup_menu(self):
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        # Load model action
        load_model_action = QAction("Load Model", self)
        load_model_action.triggered.connect(self.load_custom_model)
        file_menu.addAction(load_model_action)
        
        # Export chat history
        export_action = QAction("Export Chat History", self)
        export_action.triggered.connect(self.export_chat_history)
        file_menu.addAction(export_action)
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")
        
        # Model settings
        model_settings_action = QAction("Model Settings", self)
        model_settings_action.triggered.connect(self.show_model_settings)
        settings_menu.addAction(model_settings_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_model(self, model_path=None, tokenizer_path=None, model_size='gpt2-medium'):
        """Load the AI model"""
        try:
            self.status_bar.showMessage("Loading model...")
            QApplication.processEvents()
            
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if model_path and os.path.exists(model_path):
                # Load custom model
                self.model = GPT2LMHeadModel.from_pretrained(model_size).to(self.device)
                
                # Try loading from safetensors format first
                try:
                    from safetensors.torch import load_file
                    
                    safetensors_path = f"{model_path}/model.safetensors"
                    pytorch_path = f"{model_path}/pytorch_model.bin"
                    
                    if os.path.exists(safetensors_path):
                        state_dict = load_file(safetensors_path)
                        
                        # Check if lm_head.weight is missing and add it if needed
                        if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
                            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
                        
                        # Load state dict with strict=False to allow partial loading
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    elif os.path.exists(pytorch_path):
                        state_dict = torch.load(pytorch_path, map_location=self.device)
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    else:
                        raise FileNotFoundError(f"No model file found at {model_path}")
                        
                except Exception as e:
                    self.status_bar.showMessage(f"Error loading custom model: {str(e)}")
                    # Fall back to base model
                    self.model = GPT2LMHeadModel.from_pretrained(model_size).to(self.device)
            else:
                # Load base model
                self.model = GPT2LMHeadModel.from_pretrained(model_size).to(self.device)
            
            # Register hooks for activations
            self._register_hooks()
            
            # Update UI
            self.status_bar.showMessage(f"Model loaded: {model_size}")
            self.add_system_message("AI Assistant is ready. How can I help you today?")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def _register_hooks(self):
        """Register hooks to capture model activations"""
        # Clear existing hooks
        self._remove_hooks()
        self.activations = {}
        
        # Define hook function
        def hook_fn(name):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu().numpy()
                else:
                    self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if any(layer_type in name for layer_type in ['h.', 'mlp', 'attn']):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                
                # Add layer to the visualization combo box
                if name not in [self.viz_widget.layer_combo.itemText(i) for i in range(self.viz_widget.layer_combo.count())]:
                    self.viz_widget.layer_combo.addItem(name)
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def send_message(self):
        """Process user message and generate response"""
        user_message = self.message_input.text().strip()
        if not user_message:
            return
        
        # Clear input field
        self.message_input.clear()
        
        # Add user message to chat history
        self.add_user_message(user_message)
        
        # Handle special commands
        if user_message.lower() in ['quit', 'exit']:
            self.close()
            return
        
        if user_message.lower() == 'clear':
            self.chat_history.clear()
            self.message_history = []
            self.add_system_message("Chat history cleared.")
            return
        
        # Prepare for AI response
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.send_button.setEnabled(False)
        self.status_bar.showMessage("Generating response...")
        
        # Prepare prompt
        context = ""
        prompt = f"{context}Question: {user_message}\nAnswer:"
        
        # Create and start thread for model inference
        self.thread = ModelThread(self.model, self.tokenizer, prompt)
        self.thread.token_signal.connect(self.update_response)
        self.thread.response_signal.connect(self.save_response)
        self.thread.finished_signal.connect(self.on_generation_finished)
        self.thread.error_signal.connect(self.on_generation_error)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.start()
        
        # Prepare for streaming response
        self.current_response = ""
        self.add_ai_message_start()
    
    def update_response(self, token):
        """Update the AI response with a new token"""
        self.current_response += token
        self.update_last_message(self.current_response)
    
    def save_response(self, response):
        """Save the complete response"""
        self.complete_response = response
    
    def on_generation_finished(self):
        """Handle completion of response generation"""
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Ready")
        
        # Add the complete message to history
        self.message_history.append({"role": "assistant", "content": self.complete_response})
    
    def on_generation_error(self, error_msg):
        """Handle errors during generation"""
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.add_system_message(f"Error generating response: {error_msg}")
    
    def add_user_message(self, message):
        """Add a user message to the chat history"""
        self.chat_history.append(f"<p style='color:#0066cc'><b>You:</b> {message}</p>")
        self.message_history.append({"role": "user", "content": message})
    
    def add_ai_message_start(self):
        """Start a new AI message in the chat history"""
        self.chat_history.append(f"<p style='color:#006600'><b>AI Assistant:</b> </p>")
        # Don't add to message history yet - will be added when complete
    
    def update_last_message(self, message):
        """Update the last message in the chat history"""
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(f"<p style='color:#006600'><b>AI Assistant:</b> {message}</p>")
        self.chat_history.ensureCursorVisible()
    
    def add_system_message(self, message):
        """Add a system message to the chat history"""
        self.chat_history.append(f"<p style='color:#999999'><i>System: {message}</i></p>")
    
    def visualize_activations(self):
        """Visualize current activations"""
        if not self.activations:
            QMessageBox.warning(self, "Warning", "No activations available. Send a message first.")
            return
        
        try:
            layer_name = self.viz_widget.layer_combo.currentText()
            if layer_name == "All Layers":
                layer_name = None
            
            # Clear the figure
            self.viz_widget.figure.clear()
            
            # Create visualization
            if layer_name and layer_name in self.activations:
                # Visualize specific layer
                ax = self.viz_widget.figure.add_subplot(111)
                act = self.activations[layer_name]
                
                try:
                    if len(act.shape) > 2:
                        # For attention layers, show heatmap of first head
                        if 'attn' in layer_name:
                            try:
                                im = ax.imshow(act[0, 0], cmap='viridis')
                                ax.set_title(f"Attention Pattern: {layer_name}")
                                self.viz_widget.figure.colorbar(im, ax=ax)
                            except (IndexError, TypeError) as e:
                                # Fall back to mean activation if we can't show the first head
                                self.status_bar.showMessage(f"Error showing attention pattern: {str(e)}")
                                mean_act = np.mean(act, axis=tuple(range(len(act.shape)-1)))
                                ax.plot(mean_act)
                                ax.set_title(f"Mean Activation: {layer_name}")
                        else:
                            # For other layers with >2 dims, show mean activation
                            mean_act = np.mean(act, axis=tuple(range(len(act.shape)-1)))
                            ax.plot(mean_act)
                            ax.set_title(f"Mean Activation: {layer_name}")
                    elif len(act.shape) == 2:
                        # For 2D activations, show heatmap
                        im = ax.imshow(act, cmap='viridis')
                        ax.set_title(f"Activation: {layer_name}")
                        self.viz_widget.figure.colorbar(im, ax=ax)
                    else:
                        # For 1D activations, either plot as line or reshape to 2D
                        if act.shape[0] > 1000:
                            # Try to make a square-ish 2D array for visualization
                            side = int(np.sqrt(act.shape[0]))
                            reshaped = act[:side*side].reshape(side, side)
                            im = ax.imshow(reshaped, cmap='viridis')
                            ax.set_title(f"Activation (Reshaped): {layer_name}")
                            self.viz_widget.figure.colorbar(im, ax=ax)
                        else:
                            # Just plot as a line
                            ax.plot(act)
                            ax.set_title(f"Activation: {layer_name}")
                except Exception as e:
                    # If all else fails, just show the shape and some stats
                    self.status_bar.showMessage(f"Error visualizing layer: {str(e)}")
                    ax.text(0.5, 0.5, f"Layer: {layer_name}\nShape: {act.shape}\n"
                            f"Mean: {np.mean(act):.4f}\nStd: {np.std(act):.4f}\n"
                            f"Min: {np.min(act):.4f}\nMax: {np.max(act):.4f}",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"Layer Statistics: {layer_name}")
                    ax.axis('off')
            else:
                # Visualize summary of all layers
                try:
                    # Just show a bar chart of activation magnitudes
                    layer_names = []
                    magnitudes = []
                    
                    for name, act in self.activations.items():
                        if isinstance(act, np.ndarray):
                            layer_names.append(name.split('.')[-1])
                            magnitudes.append(np.mean(np.abs(act)))
                    
                    if layer_names:
                        ax = self.viz_widget.figure.add_subplot(111)
                        ax.bar(range(len(layer_names)), magnitudes)
                        ax.set_xticks(range(len(layer_names)))
                        ax.set_xticklabels(layer_names, rotation=90)
                        ax.set_title("Mean Activation Magnitude Across Layers")
                        ax.set_ylabel("Mean Absolute Activation")
                        self.viz_widget.figure.tight_layout()
                    else:
                        ax = self.viz_widget.figure.add_subplot(111)
                        ax.text(0.5, 0.5, "No valid activations found", 
                                ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                except Exception as e:
                    self.status_bar.showMessage(f"Error creating summary visualization: {str(e)}")
                    ax = self.viz_widget.figure.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                            ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            # Refresh canvas
            self.viz_widget.canvas.draw()
            
        except Exception as e:
            self.status_bar.showMessage(f"Visualization error: {str(e)}")
            QMessageBox.warning(self, "Visualization Error", 
                               f"An error occurred while visualizing activations:\n{str(e)}")
    
    def compare_activations(self):
        """Compare current activations with saved ones"""
        if not self.activation_history:
            QMessageBox.warning(self, "Warning", "No saved activations to compare with.")
            return
        
        # TODO: Implement comparison visualization
        QMessageBox.information(self, "Info", "Activation comparison not implemented yet.")
    
    def save_activations(self):
        """Save current activations"""
        if not self.activations:
            QMessageBox.warning(self, "Warning", "No activations to save. Send a message first.")
            return
        
        # Get label from user
        label, ok = QInputDialog.getText(self, "Save Activations", "Enter a label for these activations:")
        if ok and label:
            import copy
            self.activation_history[label] = copy.deepcopy(self.activations)
            self.status_bar.showMessage(f"Saved activations as '{label}'")
    
    def load_custom_model(self):
        """Load a custom model from file"""
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if model_dir:
            # Ask for model size
            model_sizes = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
            model_size, ok = QInputDialog.getItem(self, "Select Model Size", 
                                                "Base model size:", model_sizes, 1, False)
            if ok and model_size:
                self.load_model(model_path=model_dir, model_size=model_size)
    
    def export_chat_history(self):
        """Export chat history to a file"""
        if not self.message_history:
            QMessageBox.warning(self, "Warning", "No chat history to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Chat History", "", 
                                                "Text Files (*.txt);;JSON Files (*.json)")
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.message_history, f, indent=2)
            else:
                with open(file_path, 'w') as f:
                    for msg in self.message_history:
                        f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
            
            self.status_bar.showMessage(f"Chat history exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export chat history: {str(e)}")
    
    def show_model_settings(self):
        """Show dialog for model settings"""
        # TODO: Implement settings dialog
        QMessageBox.information(self, "Info", "Model settings dialog not implemented yet.")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About AI Chat Interface", 
                        "AI Chat Interface\n\n"
                        "A GUI application for interacting with GPT-2 based language models.\n\n"
                        "Features:\n"
                        "- Chat with AI assistant\n"
                        "- Visualize model activations\n"
                        "- Save and compare activation patterns\n")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        self._remove_hooks()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIModelChat()
    window.show()
    sys.exit(app.exec_()) 