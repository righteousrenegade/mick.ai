import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import os
import sys
import random
import datetime
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Add special tokens for conversation format
        tokenizer.pad_token = tokenizer.eos_token
        
        # Read the pre-processed training data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into individual QA pairs
        qa_pairs = text.split('\n\n')
        print(f"📊 Found {len(qa_pairs)} potential QA pairs")
        
        # Define patterns for questions and answers with simplified patterns
        question_patterns = [
            r'(?i)question[\s\:\.\-\_\*]*',  # Standard "Question:" format
            # r'(?i)q[\s\:\.\-\_\*]*(?=\s)',   # Just "Q:" at the beginning
            # r'(?i)q\d+[\s\:\.\-\_\*\)]*',    # "Q1:", "Q2:", etc.
        ]
        answer_patterns = [
            r'(?i)answer[\s\:\.\-\_\*]*',    # Standard "Answer:" format
            # r'(?i)a[\s\:\.\-\_\*]*(?=\s)',   # Just "A:" at the beginning
            # r'(?i)a\d+[\s\:\.\-\_\*\)]*',    # "A1:", "A2:", etc.
        ]
        
        # Combine patterns into regex
        question_regex = '|'.join(question_patterns)
        answer_regex = '|'.join(answer_patterns)
        
        # Filter and clean QA pairs
        processed_texts = []
        self.raw_qa_pairs = []  # Store the raw Q/A pairs for display during training
        
        for pair in qa_pairs:
            if pair.strip():  # Skip empty pairs
                # Clean up any extra whitespace or formatting
                pair = re.sub(r'\s+', ' ', pair.strip())
                
                # Assume each pair is already in "Question: ... Answer: ..." format
                # Just ensure proper spacing
                pair = re.sub(r'(Question:\s*)(.*?)(\s*Answer:\s*)', r'Question: \2 Answer: ', pair)
                
                processed_texts.append(pair)
                
                # Extract and store the question and answer separately for display
                qa_match = re.match(r'Question:\s*(.*?)\s*Answer:\s*(.*)', pair)
                if qa_match:
                    question = qa_match.group(1).strip()
                    answer = qa_match.group(2).strip()
                    self.raw_qa_pairs.append((question, answer))
                else:
                    # If we can't parse it properly, just use the whole text
                    self.raw_qa_pairs.append((pair, ""))
        
        print(f"✅ Processed {len(processed_texts)} QA pairs")
        
        if processed_texts:
            print(f"📝 Sample QA pair: {processed_texts[0][:100]}...")
        
        # Encode all texts
        self.encodings = tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        self.attention_masks = self.encodings['attention_mask']
        # Store tensors directly
        self.input_ids = self.encodings['input_ids']

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_masks[idx].clone().detach(),
            'labels': self.input_ids[idx].clone().detach(),
            'qa_pair': self.raw_qa_pairs[idx]  # Include the raw Q/A pair
        }

    def __len__(self):
        return len(self.input_ids)

def check_cuda_availability():
    """
    Comprehensive check for CUDA availability with detailed diagnostics.
    Returns a tuple of (device, is_cuda_available)
    """
    print("\n🔍 Checking CUDA availability...")
    
    # Check if CUDA is available at all
    cuda_available = torch.cuda.is_available()
    print(f"📊 torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        # Get CUDA device count
        device_count = torch.cuda.device_count()
        print(f"📊 CUDA device count: {device_count}")
        
        # Get current CUDA device
        try:
            current_device = torch.cuda.current_device()
            print(f"📊 Current CUDA device: {current_device}")
            
            # Get device name
            device_name = torch.cuda.get_device_name(current_device)
            print(f"📊 CUDA device name: {device_name}")
            
            # Get device properties
            props = torch.cuda.get_device_properties(current_device)
            print(f"📊 Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"📊 CUDA capability: {props.major}.{props.minor}")
            
            # Check memory usage
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"📊 Allocated memory: {allocated:.2f} GB")
            print(f"📊 Reserved memory: {reserved:.2f} GB")
            
            # Try a simple CUDA operation to verify it's working
            try:
                x = torch.rand(10, 10).cuda()
                y = x + x
                del x, y
                print("✅ CUDA operation successful")
                return torch.device('cuda'), True
            except Exception as e:
                print(f"❌ CUDA operation failed: {str(e)}")
                print("⚠️ Falling back to CPU")
                return torch.device('cpu'), False
        except Exception as e:
            print(f"❌ Error getting CUDA device info: {str(e)}")
            print("⚠️ Falling back to CPU")
            return torch.device('cpu'), False
    else:
        # Check for potential issues
        print("\n🔍 Diagnosing potential CUDA issues:")
        
        # Check if NVIDIA drivers are installed (for Linux/Windows)
        if sys.platform.startswith('linux'):
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    print("✅ NVIDIA drivers are installed (nvidia-smi works)")
                    print("❌ But PyTorch can't detect CUDA - possible version mismatch")
                else:
                    print("❌ NVIDIA drivers may not be installed or are not working")
            except:
                print("❌ Could not run nvidia-smi - NVIDIA drivers may not be installed")
        elif sys.platform == 'win32':
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                if result.returncode == 0:
                    print("✅ NVIDIA drivers are installed (nvidia-smi works)")
                    print("❌ But PyTorch can't detect CUDA - possible version mismatch")
                else:
                    print("❌ NVIDIA drivers may not be installed or are not working")
            except:
                print("❌ Could not run nvidia-smi - NVIDIA drivers may not be installed")
        
        # Check PyTorch build info
        print(f"📊 PyTorch version: {torch.__version__}")
        print(f"📊 PyTorch built with CUDA: {torch.version.cuda}")
        
        print("⚠️ Using CPU for training (this will be slow)")
        return torch.device('cpu'), False

def train_model(model_name="gpt2-medium", num_epochs=5, batch_size=4, learning_rate=5e-6, 
              display_qa_pairs=False, use_high_quality=True, input_file=None):
    """
    Train the model on the cleaned training data.
    
    Args:
        model_name (str): Name of the pretrained model to use
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for training
        display_qa_pairs (bool): Whether to display the QA pairs during training
        use_high_quality (bool): Whether to use the high quality dataset
        input_file (str, optional): Path to the input training file. If None, uses default based on use_high_quality.
    """
    # If input_file is provided, use it; otherwise use default based on use_high_quality
    if input_file is None:
        # Check if TRAINING_FILE environment variable is set
        import os
        input_file = os.environ.get('TRAINING_FILE')
        
        # If still None, use default based on use_high_quality
        if input_file is None:
            if use_high_quality:
                input_file = "cleaned_trainingdata_high_quality.txt"
            else:
                input_file = "cleaned_trainingdata.txt"
    
    print(f"Using training file: {input_file}")
    
    # Initialize training metrics collection
    training_metrics = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'sample_qa_pairs': [],
        'training_info': {}
    }
    
    # Record start time
    start_time = datetime.datetime.now()
    training_metrics['training_info']['start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize model and tokenizer
    print("\n🔧 Initializing model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    training_metrics['training_info']['model_name'] = model_name
    training_metrics['training_info']['model_parameters'] = sum(p.numel() for p in model.parameters())

    # Check CUDA availability with detailed diagnostics
    device, cuda_available = check_cuda_availability()
    
    training_metrics['training_info']['device'] = str(device)
    training_metrics['training_info']['cuda_available'] = cuda_available
    
    # Force CUDA if requested
    if not cuda_available and os.environ.get('FORCE_CUDA') == '1':
        print("⚠️ Forcing CUDA usage as requested by FORCE_CUDA environment variable")
        try:
            device = torch.device('cuda')
            cuda_available = True
            print("✅ Forced CUDA usage successful")
            training_metrics['training_info']['forced_cuda'] = True
        except Exception as e:
            print(f"❌ Forced CUDA usage failed: {str(e)}")
            print("⚠️ Falling back to CPU")
            device = torch.device('cpu')
            cuda_available = False
            training_metrics['training_info']['forced_cuda'] = False
            training_metrics['training_info']['cuda_error'] = str(e)

    # Prepare dataset and dataloader
    print("📚 Loading and preparing dataset...")
    full_dataset = TextDataset(input_file, tokenizer)
    
    training_metrics['training_info']['data_file'] = input_file
    training_metrics['training_info']['dataset_size'] = len(full_dataset)
    training_metrics['training_info']['use_high_quality'] = use_high_quality
    
    # Store some sample Q/A pairs for the report
    sample_indices = random.sample(range(len(full_dataset)), min(10, len(full_dataset)))
    for idx in sample_indices:
        if isinstance(full_dataset.raw_qa_pairs[idx], tuple) and len(full_dataset.raw_qa_pairs[idx]) == 2:
            question, answer = full_dataset.raw_qa_pairs[idx]
            training_metrics['sample_qa_pairs'].append({
                'question': question,
                'answer': answer
            })
    
    # Split into train and validation sets
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    training_metrics['training_info']['train_size'] = train_size
    training_metrics['training_info']['val_size'] = val_size
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Use a smaller batch size if needed to display more frequent updates
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    training_metrics['training_info']['batch_size'] = batch_size
    training_metrics['training_info']['learning_rate'] = learning_rate
    training_metrics['training_info']['num_epochs'] = num_epochs

    # Move model to the appropriate device
    model.to(device)
    print(f"📊 Model moved to device: {device}")
    
    # Verify model is on the correct device
    if cuda_available:
        print(f"📊 Model parameters on CUDA: {next(model.parameters()).is_cuda}")

    # Define optimizer and scheduler with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    try:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        training_metrics['training_info']['scheduler'] = 'cosine_with_warmup'
    except NameError:
        print("⚠️ Cosine scheduler not available, using linear scheduler")
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
        training_metrics['training_info']['scheduler'] = 'linear'

    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Track batch-level metrics
    batch_metrics = defaultdict(list)
    
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': 0,
            'val_loss': 0,
            'learning_rate': scheduler.get_last_lr()[0],
            'batches': []
        }
        
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        model.train()
        batch_count = 0
        for batch in progress_bar:
            batch_count += 1
            # Extract the Q/A pairs for display
            qa_pairs = batch.pop('qa_pair')  # Remove from batch before sending to model
            
            # Store a sample of Q/A pairs from this batch
            batch_qa_samples = []
            if batch_count % 10 == 0:  # Store samples every 10 batches to avoid too much data
                for i, qa_pair in enumerate(qa_pairs):
                    if isinstance(qa_pair, tuple) and len(qa_pair) == 2:
                        question, answer = qa_pair
                        batch_qa_samples.append({
                            'question': question[:100] + ('...' if len(question) > 100 else ''),
                            'answer': answer[:100] + ('...' if len(answer) > 100 else '')
                        })
            
            # Display the Q/A pairs being processed in this batch if enabled
            if display_qa_pairs:
                print(f"\n📝 Training on batch {batch_count}/{len(train_dataloader)} in epoch {epoch+1}")
                for i, qa_pair in enumerate(qa_pairs):
                    if isinstance(qa_pair, tuple) and len(qa_pair) == 2:
                        question, answer = qa_pair
                        print(f"\n--- Q/A Pair {i+1}/{len(qa_pairs)} ---")
                        print(f"Question: {question}")
                        print(f"Answer: {answer}")
                    else:
                        print(f"\n--- Q/A Pair {i+1}/{len(qa_pairs)} ---")
                        print(f"Q/A Pair (raw format): {qa_pair}")
                    print("-" * 50)
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Record batch metrics
            batch_loss = loss.item()
            total_train_loss += batch_loss
            batch_metrics['batch_losses'].append(batch_loss)
            batch_metrics['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Store batch info
            if batch_count % 10 == 0:  # Store detailed info every 10 batches
                epoch_metrics['batches'].append({
                    'batch_num': batch_count,
                    'loss': batch_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'qa_samples': batch_qa_samples
                })
            
            progress_bar.set_postfix(loss=batch_loss, lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_metrics['train_loss'] = avg_train_loss
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # Remove the Q/A pairs before sending to model
                batch.pop('qa_pair')
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        epoch_metrics['val_loss'] = avg_val_loss
        
        print(f'📊 Epoch {epoch+1}/{num_epochs}')
        print(f'   Training Loss: {avg_train_loss:.4f}')
        print(f'   Validation Loss: {avg_val_loss:.4f}')
        
        # Save epoch metrics
        training_metrics['epochs'].append(epoch_metrics)
        training_metrics['train_losses'].append(avg_train_loss)
        training_metrics['val_losses'].append(avg_val_loss)
        training_metrics['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"💾 Saving best model with validation loss: {best_val_loss:.4f}")
            model.save_pretrained('madison_model')
            tokenizer.save_pretrained('madison_tokenizer')
            training_metrics['training_info']['best_model_epoch'] = epoch + 1
            training_metrics['training_info']['best_val_loss'] = best_val_loss
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                training_metrics['training_info']['early_stopping'] = True
                training_metrics['training_info']['stopped_at_epoch'] = epoch + 1
                break

    # Record end time and duration
    end_time = datetime.datetime.now()
    training_metrics['training_info']['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    training_metrics['training_info']['duration_seconds'] = duration.total_seconds()
    training_metrics['training_info']['duration_formatted'] = str(duration)
    
    # Generate and save training report
    generate_training_report(training_metrics)

    print("\n✨ Training completed!")
    print(f"🏆 Best validation loss achieved: {best_val_loss:.4f}")
    print(f"📊 Training report saved to 'training_report.txt' and 'training_metrics.json'")
    return model, tokenizer

def generate_training_report(metrics):
    """Generate a comprehensive training report and save it to files."""
    # Save raw metrics as JSON for potential later analysis
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create a readable text report
    report = []
    report.append("=" * 80)
    report.append("JAImes Madison Training Report")
    report.append("=" * 80)
    
    # Training information
    info = metrics['training_info']
    report.append("\n## Training Information")
    report.append(f"Start time: {info['start_time']}")
    report.append(f"End time: {info['end_time']}")
    report.append(f"Duration: {info['duration_formatted']}")
    report.append(f"Model: {info['model_name']} ({info['model_parameters']:,} parameters)")
    report.append(f"Device: {info['device']} (CUDA: {info['cuda_available']})")
    report.append(f"Dataset: {info['data_file']} ({info['dataset_size']} examples)")
    report.append(f"Training set: {info['train_size']} examples")
    report.append(f"Validation set: {info['val_size']} examples")
    report.append(f"Batch size: {info['batch_size']}")
    report.append(f"Learning rate: {info['learning_rate']}")
    report.append(f"Scheduler: {info['scheduler']}")
    
    # Best model information
    report.append("\n## Best Model")
    report.append(f"Best epoch: {info.get('best_model_epoch', 'N/A')}")
    report.append(f"Best validation loss: {info.get('best_val_loss', 'N/A'):.6f}")
    if info.get('early_stopping', False):
        report.append(f"Early stopping triggered at epoch {info['stopped_at_epoch']}")
    
    # Training metrics summary
    report.append("\n## Training Metrics")
    report.append("Epoch | Train Loss | Val Loss | Learning Rate")
    report.append("-" * 50)
    for i, (train_loss, val_loss, lr) in enumerate(zip(
            metrics['train_losses'], 
            metrics['val_losses'], 
            metrics['learning_rates'])):
        report.append(f"{i+1:5d} | {train_loss:.6f} | {val_loss:.6f} | {lr:.8f}")
    
    # Sample Q/A pairs
    report.append("\n## Sample Q/A Pairs")
    for i, pair in enumerate(metrics['sample_qa_pairs']):
        report.append(f"\nSample {i+1}:")
        report.append(f"Q: {pair['question']}")
        report.append(f"A: {pair['answer']}")
        report.append("-" * 40)
    
    # Save the report
    with open('training_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Generate plots if matplotlib is available
    try:
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(metrics['train_losses'])+1), metrics['train_losses'], label='Training Loss')
        plt.plot(range(1, len(metrics['val_losses'])+1), metrics['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot.png')
        
        # Learning rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(metrics['learning_rates'])+1), metrics['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig('lr_plot.png')
        
        print("📊 Training plots saved to 'loss_plot.png' and 'lr_plot.png'")
    except:
        print("⚠️ Could not generate plots - matplotlib may not be installed")

# This allows the script to be run directly or imported
if __name__ == "__main__":
    # Set environment variable to force CUDA if needed
    # Uncomment the line below to force CUDA usage
    # os.environ['FORCE_CUDA'] = '1'
    
    # For best results, use these settings:
    # - Lower learning rate (5e-6 instead of 2e-5)
    # - Use the high-quality dataset
    # - More epochs (8-10) for better convergence
    train_model(
        learning_rate=5e-6,  # Lower learning rate for better quality
        num_epochs=8,        # More epochs for better convergence
        use_high_quality=True,  # Use the high-quality dataset
        display_qa_pairs=False  # Don't display Q/A pairs during training
    )