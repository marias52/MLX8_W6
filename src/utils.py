import os
import torch
import wandb

# --- Device Helper ---
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging Helper ---
def print_step(epoch, step, loss):
    print(f"✅ Epoch {epoch + 1} | Step {step:03d} | Loss: {loss:.4f}")

# --- Checkpoint Saving ---
def save_checkpoint(model, path="./checkpoints"):
    os.makedirs(path, exist_ok=True)
    model.save_lora_weights(path)
    print(f"💾 Checkpoint saved at {path}")

# --- Checkpoint Loading ---
def load_checkpoint(model_class, path):
    print(f"📂 Loading checkpoint from {path}")
    return model_class.from_pretrained_lora(path)

# --- Text Cleaning ---
def clean_text(text):
    return text.strip().replace("\n", " ")

# --- Token Count ---
def get_token_length(tokenizer, text):
    return len(tokenizer(text)["input_ids"])

# --- Weights & Biases Init ---
def init_wandb(project, name, config_dict):
    return wandb.init(project=project, name=name, config=config_dict)

# --- Log to Weights & Biases ---
def log_wandb(metrics: dict):
    wandb.log(metrics)

# --- Finish WandB ---
def finish_wandb():
    wandb.finish()


