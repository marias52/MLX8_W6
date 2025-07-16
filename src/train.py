import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from model import QwenLoRAModel  # Your model class
import pickle
from typing import Dict, List, Any
import torch.nn.functional as F
from sweep_config import sweep_configuration
import time
import traceback
import gc

# Download NLTK resources
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

print("1️⃣ Starting script")

run_type = "train"  # Change to "sweep" or "train"
print("RUNNING TYPE: ", run_type)

# --- UTILITY FUNCTIONS ---
def get_device():
    """Get the best available device with detailed info."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 CUDA available: {device_name}")
        print(f"🔥 GPU memory: {total_memory:.1f} GB")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🍎 Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("💻 Using CPU")
        return torch.device("cpu")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def save_artifact(name: str, description: str):
    """Save artifact to wandb."""
    try:
        artifact = wandb.Artifact(name=name, type="model", description=description)
        if os.path.exists('data/model.pt'):
            artifact.add_file('data/model.pt')
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"Warning: Could not save artifact: {e}")

# --- GLOBAL VARIABLES ---
device = get_device()
NUM_WORKERS = 0  # Set to 0 to avoid CUDA issues

# Create data directory
os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

print("2️⃣ Loading dataset...")
try:
    ds = load_dataset("CarperAI/openai_summarize_tldr")
    print("✅ Dataset loaded successfully")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# --- DEFAULT CONFIG ---
default_wandb_config = {
    "model_name": "Qwen/Qwen2-0.5B-Instruct",  # Smaller model to start
    "lora_rank": 8,  # Reduced for memory
    "lora_alpha": 16,  # Reduced proportionally
    "lora_dropout": 0.1,
    "learning_rate": 2e-4,
    "batch_size": 2,  # Reduced for memory
    "num_epochs": 3,
    "warmup_steps": 100,  # Reduced
    "weight_decay": 0.01,
    "max_length": 256,  # Reduced for memory
    "max_new_tokens": 64,  # Reduced for memory
    "gradient_accumulation_steps": 8,  # Increased to maintain effective batch size
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 500,
}

# --- ENHANCED QWEN MODEL CLASS ---
class EnhancedQwenLoRAModel:
    """Enhanced QwenLoRAModel with better error handling and memory management."""
    
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, max_length, device):
        self.device = device
        self.max_length = max_length
        
        try:
            print(f"🔄 Step 1: Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Tokenizer loaded successfully")
            
            print(f"🔄 Step 2: Loading base model {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                use_cache=False,  # Disable cache for training
                attn_implementation="flash_attention_2" if device.type == "cuda" else "eager"
            )
            
            print(f"✅ Base model loaded successfully")
            
            # Move to device if not using device_map
            if device.type != "cuda":
                self.model = self.model.to(device)
            
            print(f"🔄 Step 3: Setting up LoRA...")
            self._setup_lora(lora_rank, lora_alpha, lora_dropout)
            
            print(f"✅ LoRA setup complete")
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            
            print(f"🎉 Model initialization complete!")
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            traceback.print_exc()
            raise
    
    def _setup_lora(self, rank, alpha, dropout):
        """Setup LoRA configuration."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        except ImportError:
            print("❌ PEFT library not found. Please install with: pip install peft")
            raise
        except Exception as e:
            print(f"❌ LoRA setup failed: {e}")
            raise
    
    def preprocess_for_qwen(self, prompt, summary):
        """Preprocess data for Qwen model."""
        try:
            # Format the input
            input_text = f"<|im_start|>system\nYou are a helpful assistant that summarizes text.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            full_text = input_text + summary + "<|im_end|>"
            
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            
            # Create labels (only train on the assistant's response)
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            
            labels = encoding["input_ids"].clone()
            labels[0, :input_encoding["input_ids"].shape[1]] = -100  # Ignore input tokens
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            # Return dummy data to avoid crashes
            dummy_ids = torch.tensor([self.tokenizer.pad_token_id] * 10)
            return {
                "input_ids": dummy_ids,
                "attention_mask": torch.ones_like(dummy_ids),
                "labels": torch.full_like(dummy_ids, -100)
            }
    
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, do_sample=True):
        """Generate summary for a given prompt."""
        try:
            input_text = f"<|im_start|>system\nYou are a helpful assistant that summarizes text.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode only the generated part
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up the output
            generated_text = generated_text.replace("<|im_end|>", "").strip()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return "Error in generation"
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def save_lora_weights(self, path):
        """Save LoRA weights."""
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"✅ Model saved to {path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

# --- DATASET CLASSES ---
class SummarizationDataset(Dataset):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            processed = self.model.preprocess_for_qwen(item["prompt"], item["label"])
            return processed
        except Exception as e:
            print(f"❌ Error processing item {idx}: {e}")
            # Return dummy data
            dummy_ids = torch.tensor([self.model.tokenizer.pad_token_id] * 10)
            return {
                "input_ids": dummy_ids,
                "attention_mask": torch.ones_like(dummy_ids),
                "labels": torch.full_like(dummy_ids, -100)
            }

class ValDataset(Dataset):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            return {
                "prompt": item["prompt"],
                "label": item["label"]
            }
        except Exception as e:
            print(f"❌ Error processing validation item {idx}: {e}")
            return {
                "prompt": "Error",
                "label": "Error"
            }

def collate_fn(batch, tokenizer):
    """Collate function to handle batching."""
    try:
        batch_dict = {}
        for key in batch[0].keys():
            if key == "input_ids":
                batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
                    [item[key] for item in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id
                )
            elif key == "attention_mask":
                batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
                    [item[key] for item in batch],
                    batch_first=True,
                    padding_value=0
                )
            elif key == "labels":
                batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
                    [item[key] for item in batch],
                    batch_first=True,
                    padding_value=-100
                )
            else:
                batch_dict[key] = torch.stack([item[key] for item in batch])
        return batch_dict
    except Exception as e:
        print(f"❌ Error in collate_fn: {e}")
        # Return minimal batch
        return {
            "input_ids": torch.tensor([[tokenizer.pad_token_id]]),
            "attention_mask": torch.tensor([[1]]),
            "labels": torch.tensor([[-100]])
        }

# --- METEOR CALCULATION ---
def compute_meteor(model, val_loader, device, max_len=64):
    """Compute METEOR score on validation set."""
    model.model.eval()
    scores = []
    
    print("🔄 Computing METEOR scores...")
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_loader, desc="Computing METEOR")):
            if i >= 10:  # Limit to 10 samples for speed
                break
                
            try:
                prompt = sample["prompt"][0]
                actual_summary = sample["label"][0]
                
                # Generate summary
                generated_summary = model.generate(
                    prompt,
                    max_new_tokens=max_len,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Compute METEOR score
                hyp_tokens = generated_summary.split()
                ref_tokens = actual_summary.split()
                
                if len(hyp_tokens) > 0 and len(ref_tokens) > 0:
                    meteor = meteor_score([ref_tokens], hyp_tokens)
                    scores.append(meteor)
                    
            except Exception as e:
                print(f"❌ Error computing METEOR for sample {i}: {e}")
                scores.append(0.0)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"✅ Average METEOR score: {avg_score:.4f}")
    return avg_score

# --- TRAINING FUNCTION ---
def train():
    """Main training function with enhanced error handling."""
    print("4️⃣ Initializing model...")
    start_time = time.time()
    
    try:
        # Clear GPU memory before starting
        clear_gpu_memory()
        
        # Use the enhanced model class
        model = EnhancedQwenLoRAModel(
            model_name=config.model_name,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            max_length=config.max_length,
            device=device
        )
        
        print(f"✅ Model initialized in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        traceback.print_exc()
        return

    print("5️⃣ Preparing datasets...")
    
    try:
        # Create datasets with limited samples for testing
        train_subset = torch.utils.data.Subset(ds["train"], range(0, 50))  # Reduced for testing
        train_dataset = SummarizationDataset(train_subset, model)
        val_dataset = ValDataset(ds["valid"], model)
        val_subset = torch.utils.data.Subset(val_dataset, range(0, 20))  # Reduced for testing
        
        # Create collate function with tokenizer
        def collate_wrapper(batch):
            return collate_fn(batch, model.tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_wrapper,
            num_workers=NUM_WORKERS,
            pin_memory=True if device.type == "cuda" else False
        )
        
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
        
        print("✅ Dataloaders ready")
        
    except Exception as e:
        print(f"❌ Error preparing datasets: {e}")
        traceback.print_exc()
        return
    
    # Initialize optimizer and scheduler
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )
        
        print("✅ Optimizer and scheduler initialized")
        
    except Exception as e:
        print(f"❌ Error initializing optimizer: {e}")
        return
    
    global_step = 0
    best_meteor = 0.0

    for epoch in range(config.num_epochs):
        print(f"🚀 --------- Epoch {epoch + 1}/{config.num_epochs} ---------")
        
        # Create wandb table for sample predictions
        caption_table = wandb.Table(columns=["batch", "predicted_summary", "target_summary", "loss"])
        
        total_loss = 0.0
        num_batches = 0
        
        model.model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / config.gradient_accumulation_steps
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"❌ NaN loss detected at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                
                total_loss += loss.item() * config.gradient_accumulation_steps
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Clear cache periodically
                    if global_step % 10 == 0:
                        clear_gpu_memory()
                    
                    # Log training metrics
                    if global_step % config.logging_steps == 0:
                        current_loss = loss.item() * config.gradient_accumulation_steps
                        wandb.log({
                            "train_loss": current_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "global_step": global_step,
                            "epoch": epoch + 1
                        })
                
                # Log sample predictions (only first 2 batches)
                if batch_idx < 2:
                    try:
                        with torch.no_grad():
                            predicted_ids = outputs.logits.argmax(dim=-1)
                            pred_tokens = predicted_ids[0].tolist()
                            target_tokens = labels[0][labels[0] != -100].tolist()
                            
                            pred_text = model.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                            target_text = model.tokenizer.decode(target_tokens, skip_special_tokens=True)
                            
                            print(f"\n📝 [Epoch {epoch + 1} | Batch {batch_idx + 1}]")
                            print(f"Predicted: {pred_text[:100]}...")
                            print(f"Target: {target_text[:100]}...")
                            
                            caption_table.add_data(
                                batch_idx + 1,
                                pred_text[:100] + "..." if len(pred_text) > 100 else pred_text,
                                target_text[:100] + "..." if len(target_text) > 100 else target_text,
                                loss.item() * config.gradient_accumulation_steps
                            )
                    except Exception as e:
                        print(f"❌ Error logging predictions: {e}")
                        
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue
        
        # Compute average loss
        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"📊 Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
        
        # Evaluate METEOR after each epoch
        try:
            avg_meteor = compute_meteor(model, val_loader, device, config.max_new_tokens)
        except Exception as e:
            print(f"❌ Error computing METEOR: {e}")
            avg_meteor = 0.0
        
        # Save best model
        if avg_meteor > best_meteor:
            best_meteor = avg_meteor
            print(f"🎯 New best METEOR score: {best_meteor:.4f}")
            
            try:
                checkpoint_path = f"./checkpoints/best_model_epoch_{epoch + 1}"
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_lora_weights(checkpoint_path)
                
                # Save as wandb artifact
                artifact = wandb.Artifact(
                    name="best_qwen_lora_model",
                    type="model",
                    description=f"Best LoRA model with METEOR {best_meteor:.4f}"
                )
                artifact.add_dir(checkpoint_path)
                wandb.log_artifact(artifact)
                
            except Exception as e:
                print(f"❌ Error saving model: {e}")
        
        # Log epoch metrics
        try:
            log_dict = {
                "epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss,
                "meteor": avg_meteor,
                "best_meteor": best_meteor,
                f"summaries_epoch_{epoch + 1}": caption_table
            }
            
            # GPU memory logging
            if torch.cuda.is_available():
                log_dict.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3
                })
            
            wandb.log(log_dict)
            
        except Exception as e:
            print(f"❌ Error logging to wandb: {e}")
        
        # Clear memory after each epoch
        clear_gpu_memory()
    
    # Save final model
    try:
        torch.save(model.model.state_dict(), 'data/model.pt')
        save_artifact('qwen_lora_final_model', 'The final trained LoRA model for summarization')
    except Exception as e:
        print(f"❌ Error saving final model: {e}")
    
    print(f"🎉 Training completed! Best METEOR: {best_meteor:.4f}")

def main():
    """Main function to run training or sweep."""
    global config
    
    try:
        if run_type == "sweep":
            print("🔄 Running sweep mode")
            
            def train_sweep():
                global config
                wandb.init(project="QwenLoRA-Summarization")
                config = wandb.config
                train()
            
            sweep_id = wandb.sweep(sweep=sweep_configuration, project="QwenLoRA-Summarization")
            wandb.agent(
                sweep_id=sweep_id,
                function=train_sweep,
                count=10
            )
            
        elif run_type == "train":
            print("🎯 Running single training")
            wandb.init(project="QwenLoRA-Summarization", config=default_wandb_config)
            config = wandb.config
            
            print(f"📋 Config: LORA_RANK={config.lora_rank}, LORA_ALPHA={config.lora_alpha}, LR={config.learning_rate}")
            
            train()

        wandb.finish()
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()