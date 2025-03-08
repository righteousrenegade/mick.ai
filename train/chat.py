import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import List
import re
import time

class JAImesMadison:
    def __init__(self, model_path='madison_model', tokenizer_path='madison_tokenizer', model_size='gpt2-medium'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        
        # Set padding token for the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize with the correct model size
        print(f"Loading base model: {model_size}")
        self.model = GPT2LMHeadModel.from_pretrained(model_size).to(self.device)
        
        # For storing activations
        self.activations = {}
        self.hooks = []
        
        # For storing historical activations across different prompts
        self.activation_history = {}
        
        # Load your fine-tuned weights
        try:
            # Try loading from safetensors format first
            from safetensors.torch import load_file
            
            # Check if safetensors file exists
            import os
            safetensors_path = f"{model_path}/model.safetensors"
            pytorch_path = f"{model_path}/pytorch_model.bin"
            
            if os.path.exists(safetensors_path):
                print(f"Loading model from safetensors: {safetensors_path}")
                state_dict = load_file(safetensors_path)
                
                # Check if lm_head.weight is missing and add it if needed
                if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
                    print("Adding missing lm_head.weight by tying with embedding weights")
                    state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
                
                # Load state dict with strict=False to allow partial loading
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys when loading model: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in saved model: {unexpected_keys}")
                    
            elif os.path.exists(pytorch_path):
                print(f"Loading model from PyTorch binary: {pytorch_path}")
                state_dict = torch.load(pytorch_path, map_location=self.device)
                
                # Check if lm_head.weight is missing and add it if needed
                if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
                    print("Adding missing lm_head.weight by tying with embedding weights")
                    state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
                
                # Load state dict with strict=False to allow partial loading
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys when loading model: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in saved model: {unexpected_keys}")
            else:
                raise FileNotFoundError(f"Could not find model files at {model_path}")
                
        except Exception as e:
            print(f"Warning: Could not load fine-tuned weights: {str(e)}")
            print(f"Looked for model at: {model_path}/model.safetensors or {model_path}/pytorch_model.bin")
            print("Using base model instead.")
            
        self.model.eval()

    def _register_hooks(self):
        """Register hooks to capture activations from transformer layers"""
        # Clear any existing hooks
        self._remove_hooks()
        
        def hook_fn(name):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()  # Take first element if tuple
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # Register hooks for attention layers
        for i, layer in enumerate(self.model.transformer.h):
            # Capture attention outputs
            hook = layer.attn.register_forward_hook(hook_fn(f"attn_{i}"))
            self.hooks.append(hook)
            # Capture MLP outputs
            hook = layer.mlp.register_forward_hook(hook_fn(f"mlp_{i}"))
            self.hooks.append(hook)
        
        print(f"Registered {len(self.hooks)} activation hooks")
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        
    def visualize_activations(self, layer_name=None, compare_with=None):
        """
        Visualize the activations captured during inference
        
        Args:
            layer_name: Optional specific layer to visualize
            compare_with: Optional list of labels from activation_history to compare with
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        if not self.activations:
            print("No activations captured. Run inference first.")
            return
        
        # If comparing with historical activations
        if compare_with and isinstance(compare_with, list):
            valid_labels = [label for label in compare_with if label in self.activation_history]
            
            if not valid_labels:
                print("No valid comparison labels found in history.")
                return
                
            print(f"Comparing current activations with: {valid_labels}")
            
            # For specific layer comparison
            if layer_name and layer_name in self.activations:
                self._compare_specific_layer(layer_name, valid_labels)
            else:
                # Compare average activation magnitudes across all layers
                self._compare_all_layers(valid_labels)
                
        # Regular single activation visualization
        elif layer_name and layer_name in self.activations:
            # Visualize specific layer
            act = self.activations[layer_name]
            print(f"Visualizing {layer_name} activation shape: {act.shape}")
            self._plot_activation(act, layer_name)
        else:
            print(f"Visualizing all activations")
            # Show a summary of all activations
            plt.figure(figsize=(12, 8))
            
            # Sort layer names to get a consistent order
            layer_names = sorted(self.activations.keys())
            
            # Calculate average activation magnitude for each layer
            magnitudes = []
            for name in layer_names:
                act = self.activations[name]
                if isinstance(act, tuple):
                    act = act[0]  # Some layers return tuples
                # Calculate mean absolute value
                mag = act.abs().mean().cpu().numpy()
                magnitudes.append(mag)
            
            plt.bar(range(len(magnitudes)), magnitudes)
            plt.xticks(range(len(magnitudes)), layer_names, rotation=90)
            plt.title("Average Activation Magnitude by Layer")
            plt.tight_layout()
            plt.show()
    
    def _compare_all_layers(self, labels):
        """Compare activation magnitudes across all layers for multiple inputs"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get current activation magnitudes
        current_magnitudes = {}
        for name in sorted(self.activations.keys()):
            act = self.activations[name]
            if isinstance(act, tuple):
                act = act[0]
            current_magnitudes[name] = act.abs().mean().cpu().numpy()
        
        # Get historical activation magnitudes
        historical_magnitudes = {}
        for label in labels:
            historical_magnitudes[label] = {}
            for name in sorted(self.activation_history[label].keys()):
                act = self.activation_history[label][name]
                if isinstance(act, tuple):
                    act = act[0]
                historical_magnitudes[label][name] = act.abs().mean().cpu().numpy()
        
        # Create comparison plot
        plt.figure(figsize=(15, 10))
        
        # Plot current activations
        layer_names = sorted(current_magnitudes.keys())
        x = np.arange(len(layer_names))
        width = 0.8 / (len(labels) + 1)  # Bar width
        
        # Plot current activations
        plt.bar(x, [current_magnitudes[name] for name in layer_names], 
                width=width, label='Current')
        
        # Plot historical activations
        for i, label in enumerate(labels):
            offset = (i + 1) * width
            values = [historical_magnitudes[label].get(name, 0) for name in layer_names]
            plt.bar(x + offset, values, width=width, label=label)
        
        plt.xlabel('Layer')
        plt.ylabel('Average Activation Magnitude')
        plt.title('Activation Comparison Across Inputs')
        plt.xticks(x + width * (len(labels)) / 2, layer_names, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Also show difference heatmap
        self._plot_activation_differences(current_magnitudes, historical_magnitudes, layer_names)
    
    def _plot_activation_differences(self, current, historical, layer_names):
        """Plot a heatmap of activation differences between inputs"""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        # Create a matrix of differences
        labels = list(historical.keys())
        diff_matrix = np.zeros((len(labels), len(layer_names)))
        
        for i, label in enumerate(labels):
            for j, layer in enumerate(layer_names):
                # Calculate relative difference
                current_val = current[layer]
                hist_val = historical[label].get(layer, 0)
                
                if current_val != 0 or hist_val != 0:
                    # Use relative difference
                    avg = (current_val + hist_val) / 2
                    if avg != 0:
                        diff_matrix[i, j] = (current_val - hist_val) / avg
                    else:
                        diff_matrix[i, j] = 0
                else:
                    diff_matrix[i, j] = 0
        
        # Plot heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(diff_matrix, cmap='coolwarm', center=0,
                   xticklabels=layer_names, yticklabels=labels)
        plt.title('Relative Activation Differences (Red = Current Higher, Blue = Historical Higher)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    def _compare_specific_layer(self, layer_name, labels):
        """Compare a specific layer's activations across different inputs"""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        # Get current activation
        current_act = self.activations[layer_name]
        if isinstance(current_act, tuple):
            current_act = current_act[0]
        
        # Get historical activations for this layer
        historical_acts = {}
        for label in labels:
            if layer_name in self.activation_history[label]:
                act = self.activation_history[label][layer_name]
                if isinstance(act, tuple):
                    act = act[0]
                historical_acts[label] = act
        
        # For attention layers (4D tensors)
        if len(current_act.shape) == 4:
            # Compare attention patterns across different heads
            self._compare_attention_patterns(current_act, historical_acts, layer_name)
        else:
            # For other activations, compare activation patterns
            self._compare_activation_patterns(current_act, historical_acts, layer_name)
    
    def _compare_attention_patterns(self, current_act, historical_acts, layer_name):
        """Compare attention patterns across different inputs"""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        # Get number of attention heads
        num_heads = min(4, current_act.shape[1])  # Limit to 4 heads for clarity
        
        # Create a grid of plots
        fig, axes = plt.subplots(len(historical_acts) + 1, num_heads, 
                                figsize=(4 * num_heads, 3 * (len(historical_acts) + 1)))
        
        # If only one head, make sure axes is 2D
        if num_heads == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot current attention patterns
        for h in range(num_heads):
            sns.heatmap(current_act[0, h].cpu().numpy(), ax=axes[0, h], cmap="viridis")
            axes[0, h].set_title(f"Current: Head {h}")
            axes[0, h].axis('off')
        
        # Plot historical attention patterns
        for i, (label, act) in enumerate(historical_acts.items(), 1):
            for h in range(num_heads):
                sns.heatmap(act[0, h].cpu().numpy(), ax=axes[i, h], cmap="viridis")
                axes[i, h].set_title(f"{label}: Head {h}")
                axes[i, h].axis('off')
        
        plt.suptitle(f"Attention Pattern Comparison for {layer_name}")
        plt.tight_layout()
        plt.show()
    
    def _compare_activation_patterns(self, current_act, historical_acts, layer_name):
        """Compare activation patterns for non-attention layers"""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        # Number of plots
        n_plots = len(historical_acts) + 1
        
        # Create figure
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]  # Make sure axes is iterable
        
        # Process current activation
        current_np = current_act.cpu().numpy()
        if len(current_np.shape) > 2:
            current_np = current_np[0]  # Take first batch item
            if len(current_np.shape) > 2:
                seq_len = current_np.shape[0]
                current_np = current_np.reshape(seq_len, -1)
        
        # Plot current activation
        sns.heatmap(current_np, ax=axes[0], cmap="viridis")
        axes[0].set_title(f"Current: {layer_name}")
        
        # Plot historical activations
        for i, (label, act) in enumerate(historical_acts.items(), 1):
            hist_np = act.cpu().numpy()
            if len(hist_np.shape) > 2:
                hist_np = hist_np[0]  # Take first batch item
                if len(hist_np.shape) > 2:
                    seq_len = hist_np.shape[0]
                    hist_np = hist_np.reshape(seq_len, -1)
            
            sns.heatmap(hist_np, ax=axes[i], cmap="viridis")
            axes[i].set_title(f"{label}: {layer_name}")
        
        plt.tight_layout()
        plt.show()
        
        # Also show difference heatmap for the first historical activation
        if historical_acts:
            first_label = list(historical_acts.keys())[0]
            first_act = historical_acts[first_label].cpu().numpy()
            
            if len(first_act.shape) > 2:
                first_act = first_act[0]
                if len(first_act.shape) > 2:
                    seq_len = first_act.shape[0]
                    first_act = first_act.reshape(seq_len, -1)
            
            # Calculate difference
            if current_np.shape == first_act.shape:
                plt.figure(figsize=(10, 6))
                diff = current_np - first_act
                sns.heatmap(diff, cmap='coolwarm', center=0)
                plt.title(f"Activation Difference: Current vs {first_label}")
                plt.tight_layout()
                plt.show()

    def list_saved_activations(self):
        """List all saved activation labels"""
        if not self.activation_history:
            print("No saved activations found.")
            return
        
        print("Saved activation labels:")
        for i, label in enumerate(self.activation_history.keys(), 1):
            print(f"{i}. {label}")
    
    def clear_activation_history(self):
        """Clear the activation history"""
        self.activation_history = {}
        print("Activation history cleared.")

    def _generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7, 
                           save_activations=True, activation_label=None) -> str:
        """Generate a response with the model and stream it token by token"""
        # Register hooks to capture activations
        self._register_hooks()
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        try:
            # First generate the full response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length + len(input_ids[0]),
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    # Add these parameters to help with complete sentences
                    min_length=len(input_ids[0]) + 10,  # Ensure some minimum response length
                    no_repeat_ngram_size=3,  # Avoid repetition
                    repetition_penalty=1.2,  # Penalize repetition
                )
            
            # Get the full generated text
            full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the response part (after the prompt)
            response = full_text[len(prompt):].strip()
            
            # Now stream the response token by token for visual effect
            print("\nJAImes Madison: ", end="", flush=True)
            
            # Simulate streaming by printing character by character
            for char in response:
                print(char, end="", flush=True)
                # Optional: add a small delay to make the streaming more visible
                time.sleep(0.01)  # 10ms delay between characters
            
            print()  # Add a newline at the end
            
            # Save activations if requested
            if save_activations and activation_label:
                # Create a deep copy of activations to store
                import copy
                self.activation_history[activation_label] = copy.deepcopy(self.activations)
                print(f"Saved activations with label: {activation_label}")
            
            return response
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            return ""

    def _format_response(self, text: str) -> str:
        # Add proper punctuation and formatting
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = text.capitalize()
        if not text.endswith(('.', '!', '?')):
            text += '.'
        # Clean up multiple spaces
        text = ' '.join(text.split())
        return text

    def _ensure_complete_sentence(self, text):
        """Ensure the text ends with a complete sentence"""
        # If text doesn't end with sentence-ending punctuation
        if not re.search(r'[.!?]\s*$', text):
            # Find the last sentence-ending punctuation
            match = re.search(r'(.*[.!?])\s+[^.!?]*$', text)
            if match:
                # Return text up to the last complete sentence
                return match.group(1)
        return text

    def chat(self):
        print("Welcome! You are now chatting with JAImes Madison, the AI version of James Madison.")
        print("(Type 'quit', 'exit' to end the conversation, or 'viz' to visualize activations)\n")
        print("Advanced commands:")
        print("  'save [label]' - Save current activations with a label")
        print("  'list' - List all saved activation labels")
        print("  'compare [label1] [label2]...' - Compare current activations with saved ones")
        print("  'patterns' - Visualize patterns across all saved activations")
        print("  'patterns [directory]' - Save visualizations to specified directory")
        print("  'clear' - Clear activation history")
        
        context = "As James Madison, I shall respond to your inquiries with the wisdom of the Federalist Papers. "
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nJAImes Madison: Farewell, and may the principles of federalism guide your path.")
                break
            
            # Handle visualization commands
            if user_input.lower() == 'viz':
                print("Visualizing all activations")
                self.visualize_activations()
                continue
                
            if user_input.lower().startswith('viz '):
                print(f"Visualizing {user_input[4:].strip()}")
                layer_name = user_input[4:].strip()
                self.visualize_activations(layer_name)
                continue
            
            # Handle activation saving
            if user_input.lower().startswith('save '):
                label = user_input[5:].strip()
                if not label:
                    print("Please provide a label to save the activations.")
                    continue
                
                if not self.activations:
                    print("No activations to save. Ask a question first.")
                    continue
                
                import copy
                self.activation_history[label] = copy.deepcopy(self.activations)
                print(f"Saved current activations as '{label}'")
                continue
            
            # List saved activations
            if user_input.lower() == 'list':
                self.list_saved_activations()
                continue
            
            # Clear activation history
            if user_input.lower() == 'clear':
                self.clear_activation_history()
                continue
            
            # Compare activations
            if user_input.lower().startswith('compare '):
                parts = user_input[8:].strip().split()
                if not parts:
                    print("Please specify which saved activations to compare with.")
                    continue
                
                if not self.activations:
                    print("No current activations. Ask a question first.")
                    continue
                
                # Check if comparing a specific layer
                if len(parts) >= 2 and parts[0] == 'layer':
                    layer_name = parts[1]
                    labels = parts[2:]
                    if not labels:
                        print("Please specify which saved activations to compare with.")
                        continue
                    
                    if layer_name not in self.activations:
                        print(f"Layer {layer_name} not found in current activations.")
                        continue
                    
                    self.visualize_activations(layer_name, compare_with=labels)
                else:
                    # Compare all layers
                    self.visualize_activations(compare_with=parts)
                continue
            
            # Visualize patterns across saved activations
            if user_input.lower() == 'patterns':
                self.visualize_patterns()
                continue
            
            # Allow specifying a custom output directory
            if user_input.lower().startswith('patterns '):
                output_dir = user_input[9:].strip()
                self.visualize_patterns(output_dir=output_dir)
                continue
            
            # Prepare prompt with context
            prompt = f"{context}Question: {user_input}\nAnswer:"
            
            # Generate response (now with streaming)
            response = self._generate_response(prompt)
            
            # No need to print the formatted response as it's already streamed
            # But we can still format it for internal use if needed
            formatted_response = self._format_response(response)

    def visualize_patterns(self, output_dir="visualizations"):
        """Visualize patterns across saved activations and save to directory"""
        if not self.activation_history or len(self.activation_history) < 2:
            print("Need at least 2 saved activation sets to visualize patterns.")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from scipy.cluster.hierarchy import dendrogram, linkage
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Create a timestamped subdirectory for this analysis session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, f"analysis_{timestamp}")
        os.makedirs(session_dir)
        print(f"Saving visualizations to: {session_dir}")
        
        # Create a log file with analysis details
        with open(os.path.join(session_dir, "analysis_log.txt"), "w") as log_file:
            log_file.write(f"Activation Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Prompts analyzed: {', '.join(self.activation_history.keys())}\n\n")
        
        print("Analyzing activation patterns across saved prompts...")
        
        # Extract activation magnitudes for each layer across all saved prompts
        layer_names = sorted(next(iter(self.activation_history.values())).keys())
        prompt_labels = list(self.activation_history.keys())
        
        # Create a matrix: rows=prompts, columns=layers
        activation_matrix = np.zeros((len(prompt_labels), len(layer_names)))
        
        for i, label in enumerate(prompt_labels):
            for j, layer in enumerate(layer_names):
                if layer in self.activation_history[label]:
                    act = self.activation_history[label][layer]
                    if isinstance(act, tuple):
                        act = act[0]
                    activation_matrix[i, j] = act.abs().mean().cpu().numpy()
        
        # 1. Correlation matrix between prompts
        plt.figure(figsize=(10, 8))
        prompt_corr = np.corrcoef(activation_matrix)
        sns.heatmap(prompt_corr, annot=True, cmap='coolwarm', 
                   xticklabels=prompt_labels, yticklabels=prompt_labels)
        plt.title('Correlation Between Different Prompts')
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, "01_prompt_correlation.png"), dpi=300)
        plt.close()
        print("✓ Saved prompt correlation matrix")
        
        # 2. Correlation matrix between layers
        plt.figure(figsize=(16, 14))
        layer_corr = np.corrcoef(activation_matrix.T)
        
        # Use a mask to highlight the most correlated pairs
        mask = np.zeros_like(layer_corr)
        mask[np.triu_indices_from(mask)] = True
        
        # Plot with a diverging colormap
        sns.heatmap(layer_corr, mask=mask, cmap='coolwarm', center=0,
                   xticklabels=layer_names, yticklabels=layer_names)
        plt.title('Correlation Between Different Layers')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, "02_layer_correlation.png"), dpi=300)
        plt.close()
        print("✓ Saved layer correlation matrix")
        
        # 3. Hierarchical clustering of layers
        plt.figure(figsize=(14, 8))
        
        # Compute linkage matrix
        Z = linkage(activation_matrix.T, 'ward')
        
        # Plot dendrogram
        dendrogram(Z, labels=layer_names, leaf_rotation=90)
        plt.title('Hierarchical Clustering of Layers')
        plt.xlabel('Layers')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, "03_layer_clustering.png"), dpi=300)
        plt.close()
        print("✓ Saved hierarchical clustering dendrogram")
        
        # 4. t-SNE visualization of layers
        if len(layer_names) > 3:  # Only do t-SNE if we have enough layers
            # Apply t-SNE to reduce dimensionality to 2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(layer_names)-1))
            layer_tsne = tsne.fit_transform(activation_matrix.T)
            
            # Create a scatter plot
            plt.figure(figsize=(12, 10))
            
            # Separate attention and MLP layers for coloring
            attn_indices = [i for i, name in enumerate(layer_names) if 'attn' in name]
            mlp_indices = [i for i, name in enumerate(layer_names) if 'mlp' in name]
            
            # Plot points
            plt.scatter(layer_tsne[attn_indices, 0], layer_tsne[attn_indices, 1], 
                       c='blue', label='Attention Layers', s=100, alpha=0.7)
            plt.scatter(layer_tsne[mlp_indices, 0], layer_tsne[mlp_indices, 1], 
                       c='red', label='MLP Layers', s=100, alpha=0.7)
            
            # Add labels to each point
            for i, name in enumerate(layer_names):
                plt.annotate(name, (layer_tsne[i, 0], layer_tsne[i, 1]), 
                            fontsize=9, alpha=0.8)
            
            plt.title('t-SNE Visualization of Layer Activations')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(session_dir, "04_tsne_visualization.png"), dpi=300)
            plt.close()
            print("✓ Saved t-SNE visualization")
        
        # 5. Heatmap of activations across prompts and layers
        # Normalize the data for better visualization
        normalized_matrix = activation_matrix / activation_matrix.max(axis=0)
        
        # Create a custom diverging colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Plot heatmap with both row and column clustering
        g = sns.clustermap(normalized_matrix, 
                         cmap=cmap,
                         row_cluster=True, 
                         col_cluster=True,
                         xticklabels=layer_names,
                         yticklabels=prompt_labels,
                         figsize=(16, 10),
                         cbar_pos=(0.02, 0.8, 0.05, 0.18))
        
        # Rotate column labels
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
        plt.suptitle('Clustered Heatmap of Layer Activations Across Prompts', y=0.95, fontsize=16)
        plt.savefig(os.path.join(session_dir, "05_clustered_heatmap.png"), dpi=300)
        plt.close()
        print("✓ Saved clustered heatmap")
        
        # 6. Radar chart for selected layers
        # Choose a subset of important layers based on variance
        layer_variance = np.var(normalized_matrix, axis=0)
        top_indices = np.argsort(layer_variance)[-min(8, len(layer_names)):]
        selected_layers = [layer_names[i] for i in top_indices]
        
        # Create radar chart
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        def radar_factory(num_vars, frame='circle'):
            # Calculate evenly-spaced axis angles
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            
            # Rotate theta such that the first axis is at the top
            theta += np.pi/2
            
            def draw_poly_patch(ax):
                # Draw polygon connecting the axis lines
                verts = unit_poly_verts(theta)
                return plt.Polygon(verts, closed=True, edgecolor='k')
            
            def draw_circle_patch(ax):
                # Draw circular patch (background)
                return plt.Circle((0.5, 0.5), 0.5)
            
            def unit_poly_verts(theta):
                # Return vertices of polygon for subplot axes
                x0, y0, r = [0.5] * 3
                verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
                return verts
            
            patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
            if frame not in patch_dict:
                raise ValueError('unknown value for `frame`: %s' % frame)
            
            class RadarAxes(plt.PolarAxes):
                name = 'radar'
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                
                def fill(self, *args, **kwargs):
                    """Override fill so that line is closed by default"""
                    closed = kwargs.pop('closed', True)
                    return super().fill(closed=closed, *args, **kwargs)
                
                def plot(self, *args, **kwargs):
                    """Override plot so that line is closed by default"""
                    lines = super().plot(*args, **kwargs)
                    for line in lines:
                        self._close_line(line)
                
                def _close_line(self, line):
                    x, y = line.get_data()
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)
                
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)
                
                def _gen_axes_patch(self):
                    return patch_dict[frame](self)
            
            register_projection(RadarAxes)
            return theta
        
        from matplotlib.projections import register_projection
        
        # Create the radar chart
        if selected_layers:
            theta = radar_factory(len(selected_layers), frame='polygon')
            
            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
            
            # Plot each prompt
            for i, label in enumerate(prompt_labels):
                values = [normalized_matrix[i, layer_names.index(layer)] for layer in selected_layers]
                ax.plot(theta, values, 'o-', linewidth=2, label=label)
                ax.fill(theta, values, alpha=0.1)
            
            ax.set_varlabels(selected_layers)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Radar Chart of Top Variable Layers Across Prompts', size=15)
            plt.tight_layout()
            plt.savefig(os.path.join(session_dir, "06_radar_chart.png"), dpi=300)
            plt.close()
            print("✓ Saved radar chart")
        
        # 7. Generate an HTML report with all visualizations
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Activation Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                .image-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
                .description {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Neural Network Activation Analysis</h1>
            <p>Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Prompts analyzed: {', '.join(prompt_labels)}</p>
            
            <h2>1. Correlation Between Prompts</h2>
            <div class="description">
                <p>This visualization shows how similar different prompts are in terms of their activation patterns across all layers. 
                High correlation (bright red) indicates prompts that activate the model in similar ways.</p>
            </div>
            <div class="image-container">
                <img src="01_prompt_correlation.png" alt="Prompt Correlation Matrix">
            </div>
            
            <h2>2. Correlation Between Layers</h2>
            <div class="description">
                <p>This visualization reveals which layers tend to activate together across different prompts. 
                Clusters of correlated layers likely form functional units within the network.</p>
            </div>
            <div class="image-container">
                <img src="02_layer_correlation.png" alt="Layer Correlation Matrix">
            </div>
            
            <h2>3. Hierarchical Clustering of Layers</h2>
            <div class="description">
                <p>This dendrogram groups layers based on similarity in their activation patterns.
                Layers that branch together at lower heights are more functionally similar.</p>
            </div>
            <div class="image-container">
                <img src="03_layer_clustering.png" alt="Hierarchical Clustering Dendrogram">
            </div>
            
            <h2>4. t-SNE Visualization</h2>
            <div class="description">
                <p>This 2D mapping positions similar layers closer together, with attention layers in blue and MLP layers in red.
                Look for unexpected groupings that might reveal specialized functions.</p>
            </div>
            <div class="image-container">
                <img src="04_tsne_visualization.png" alt="t-SNE Visualization">
            </div>
            
            <h2>5. Clustered Heatmap</h2>
            <div class="description">
                <p>This heatmap shows normalized activation strengths for each layer (columns) across different prompts (rows).
                Both rows and columns are clustered to reveal patterns of specialization.</p>
            </div>
            <div class="image-container">
                <img src="05_clustered_heatmap.png" alt="Clustered Heatmap">
            </div>
            
            <h2>6. Radar Chart of Key Layers</h2>
            <div class="description">
                <p>This radar chart highlights the most variable layers across different prompts.
                Each prompt creates a distinctive "fingerprint" that reveals which aspects of the model it activates.</p>
            </div>
            <div class="image-container">
                <img src="06_radar_chart.png" alt="Radar Chart">
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(session_dir, "analysis_report.html"), "w") as f:
            f.write(html_content)
        
        print(f"✓ Generated HTML report")
        print(f"\nAnalysis complete! All visualizations saved to: {session_dir}")
        print(f"Open {os.path.join(session_dir, 'analysis_report.html')} in a browser to view the report.")
        
        return session_dir

if __name__ == "__main__":
    try:
        # Try with GPT-2 Medium since that matches the dimensions in the error
        madison = JAImesMadison(model_size='gpt2-medium')
        madison.chat()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure the model and tokenizer files are in the correct location.") 