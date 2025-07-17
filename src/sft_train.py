import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
CONFIG = {
    "model_name": "Qwen/Qwen2-1.5B-Instruct",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 1e-4,
    "batch_size": 1,
    "num_epochs": 2,
    "max_length": 128,
    "max_new_tokens": 32,
    "gradient_accumulation_steps": 4,
}

# Dataset Class
class SummarizationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            prompt = item.get("prompt", "")
            summary = item.get("label", "")
            
            if not prompt or not summary:
                return self._get_dummy_item()
            
            # Simple format
            input_text = f"Summarize: {prompt[:300]}\nSummary:"
            full_text = input_text + " " + summary[:100]
            
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
                add_special_tokens=True
            )
            
            # Create labels
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
                add_special_tokens=True
            )
            
            labels = encoding["input_ids"].clone()
            prompt_len = input_encoding["input_ids"].shape[1]
            
            if prompt_len < labels.shape[1]:
                labels[:, :prompt_len] = -100
            else:
                labels[:, :] = -100
            
            # Clamp to vocab size
            input_ids = torch.clamp(encoding["input_ids"], 0, self.vocab_size - 1)
            labels = torch.clamp(labels, -100, self.vocab_size - 1)
            
            return {
                "input_ids": input_ids.squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return self._get_dummy_item()
    
    def _get_dummy_item(self):
        dummy_len = 16
        dummy_ids = torch.full((dummy_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        return {
            "input_ids": dummy_ids,
            "attention_mask": torch.ones_like(dummy_ids),
            "labels": torch.full_like(dummy_ids, -100)
        }

def train():
    """Main training function"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("CarperAI/openai_summarize_tldr")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        use_cache=False,
        attn_implementation="eager",
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Create datasets
    print("Creating datasets...")
    train_subset = torch.utils.data.Subset(ds["train"], range(0, 20))  # Small subset for testing
    train_dataset = SummarizationDataset(train_subset, tokenizer, CONFIG["max_length"])
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=True
    )
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG["learning_rate"])
    
    print("Starting training...")
    model.train()
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            try:
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Clamp tokens to vocab size
                vocab_size = model.config.vocab_size
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                labels = torch.where(labels == -100, labels, torch.clamp(labels, 0, vocab_size - 1))
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / CONFIG["gradient_accumulation_steps"]
                
                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() <= 0:
                    print(f"Invalid loss at batch {batch_idx}: {loss.item()}, skipping...")
                    continue
                
                loss.backward()
                total_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
                num_batches += 1
                
                # Update weights
                if (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Log progress
                if batch_idx % 5 == 0:
                    print(f"Batch {batch_idx}: Loss = {loss.item() * CONFIG['gradient_accumulation_steps']:.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    print("Saving model...")
    os.makedirs("./checkpoints", exist_ok=True)
    model.save_pretrained("./checkpoints/final_model")
    tokenizer.save_pretrained("./checkpoints/final_model")
    
    print("Training completed!")

if __name__ == "__main__":
    train()