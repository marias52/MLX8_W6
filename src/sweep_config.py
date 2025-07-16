# sweep_config.py
"""
Configuration for wandb sweeps and utility functions.
"""

import wandb
import os
import torch

# --- SWEEP CONFIGURATIONS ---

# Main sweep configuration (used in training script)
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'meteor',
        'goal': 'maximize'
    },
    'parameters': {
        'lora_rank': {
            'values': [8, 16, 32, 64]
        },
        'lora_alpha': {
            'values': [16, 32, 64, 128]
        },
        'lora_dropout': {
            'min': 0.05,
            'max': 0.3
        },
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [4, 8, 16]
        },
        'warmup_steps': {
            'values': [100, 200, 500]
        },
        'weight_decay': {
            'min': 0.0,
            'max': 0.1
        }
    }
}

# Basic LoRA hyperparameter sweep
lora_sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_meteor',
        'goal': 'maximize'
    },
    'parameters': {
        'lora_rank': {
            'values': [8, 16, 32, 64]
        },
        'lora_alpha': {
            'values': [16, 32, 64, 128]
        },
        'lora_dropout': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.3
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [4, 8, 16, 32]
        },
        'warmup_steps': {
            'values': [100, 200, 500, 1000]
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.1
        }
    }
}

# Advanced sweep with different model sizes
model_size_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_meteor',
        'goal': 'maximize'
    },
    'parameters': {
        'model_name': {
            'values': [
                "Qwen/Qwen2-0.5B-Instruct",
                "Qwen/Qwen2-1.5B-Instruct",
                "Qwen/Qwen2-7B-Instruct"
            ]
        },
        'lora_rank': {
            'values': [16, 32]
        },
        'lora_alpha': {
            'values': [32, 64]
        },
        'learning_rate': {
            'values': [1e-4, 2e-4, 5e-4]
        },
        'batch_size': {
            'values': [4, 8, 16]
        }
    }
}

# Learning rate and scheduler sweep
lr_scheduler_sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_meteor',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-3
        },
        'warmup_steps': {
            'values': [50, 100, 200, 500]
        },
        'scheduler_type': {
            'values': ['linear', 'cosine', 'constant']
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.2
        },
        'gradient_accumulation_steps': {
            'values': [1, 2, 4, 8]
        }
    }
}

# Target modules sweep for LoRA
target_modules_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_meteor',
        'goal': 'maximize'
    },
    'parameters': {
        'target_modules_config': {
            'values': [
                'attention_only',  # Only attention layers
                'mlp_only',        # Only MLP layers
                'all_linear',      # All linear layers
                'custom_1',        # Custom combination 1
                'custom_2'         # Custom combination 2
            ]
        },
        'lora_rank': {
            'values': [16, 32]
        },
        'lora_alpha': {
            'values': [32, 64]
        },
        'learning_rate': {
            'values': [1e-4, 2e-4]
        }
    }
}

# --- SWEEP CONFIGURATIONS REGISTRY ---
SWEEP_CONFIGS = {
    'lora_basic': lora_sweep_config,
    'model_size': model_size_sweep_config,
    'lr_scheduler': lr_scheduler_sweep_config,
    'target_modules': target_modules_sweep_config
}

# --- UTILITY FUNCTIONS ---

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def init_wandb(project_name: str = "qwen-lora-summarization", entity: str = None):
    """Initialize wandb with proper configuration."""
    wandb.login()
    
    # Set up wandb environment
    os.environ["WANDB_PROJECT"] = project_name
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    
    return wandb

def save_artifact(name: str, description: str, artifact_type: str = "model", file_path: str = None):
    """Save artifact to wandb."""
    artifact = wandb.Artifact(name=name, type=artifact_type, description=description)
    
    if file_path:
        if os.path.isdir(file_path):
            artifact.add_dir(file_path)
        else:
            artifact.add_file(file_path)
    
    wandb.log_artifact(artifact)
    return artifact

def load_artifact_path(artifact_name: str, version: str = "latest", file_extension: str = None):
    """Load artifact from wandb and return local path."""
    artifact = wandb.use_artifact(f"{artifact_name}:{version}")
    artifact_dir = artifact.download()
    
    if file_extension:
        # Find file with specific extension
        for file in os.listdir(artifact_dir):
            if file.endswith(file_extension):
                return os.path.join(artifact_dir, file)
    
    return artifact_dir

def get_target_modules_config(config_name: str):
    """Get target modules configuration based on config name."""
    target_modules_configs = {
        'attention_only': ["q_proj", "k_proj", "v_proj", "o_proj"],
        'mlp_only': ["gate_proj", "up_proj", "down_proj"],
        'all_linear': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'custom_1': ["q_proj", "v_proj", "gate_proj", "down_proj"],
        'custom_2': ["q_proj", "k_proj", "v_proj", "up_proj"]
    }
    
    return target_modules_configs.get(config_name, target_modules_configs['all_linear'])

def create_run_name(config):
    """Create a descriptive run name based on config."""
    model_name = config.get('model_name', 'qwen').split('/')[-1]
    lora_rank = config.get('lora_rank', 16)
    lora_alpha = config.get('lora_alpha', 32)
    lr = config.get('learning_rate', 2e-4)
    batch_size = config.get('batch_size', 8)
    
    return f"{model_name}_r{lora_rank}_a{lora_alpha}_lr{lr:.0e}_bs{batch_size}"

def log_model_info(model, config):
    """Log model information to wandb."""
    # Get model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_info = {
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
        }
    else:
        memory_info = {"gpu_memory_allocated_gb": 0}
    
    # Log to wandb
    wandb.log({
        "model_info/total_parameters": total_params,
        "model_info/trainable_parameters": trainable_params,
        "model_info/trainable_percentage": 100 * trainable_params / total_params,
        "model_info/model_size_mb": total_params * 4 / 1024**2,  # Assuming float32
        **memory_info
    })
    
    # Log config as table
    config_table = wandb.Table(columns=["Parameter", "Value"])
    for key, value in config.items():
        config_table.add_data(key, str(value))
    
    wandb.log({"model_config": config_table})

def log_training_samples(model, tokenizer, samples, step, max_samples=5):
    """Log training samples to wandb."""
    table = wandb.Table(columns=["Step", "Prompt", "Generated", "Target", "Prompt_Length", "Generated_Length"])
    
    for i, sample in enumerate(samples[:max_samples]):
        prompt = sample.get('prompt', '')
        generated = sample.get('generated', '')
        target = sample.get('target', '')
        
        table.add_data(
            step,
            prompt[:200] + "..." if len(prompt) > 200 else prompt,
            generated[:200] + "..." if len(generated) > 200 else generated,
            target[:200] + "..." if len(target) > 200 else target,
            len(prompt.split()),
            len(generated.split())
        )
    
    wandb.log({f"training_samples_step_{step}": table})

def setup_sweep(sweep_type: str, project_name: str, count: int = 10):
    """Setup and run a wandb sweep."""
    if sweep_type not in SWEEP_CONFIGS:
        print(f"Available sweep types: {list(SWEEP_CONFIGS.keys())}")
        raise ValueError(f"Unknown sweep type: {sweep_type}")
    
    sweep_config = SWEEP_CONFIGS[sweep_type]
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name
    )
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Sweep configuration: {sweep_config}")
    print(f"View sweep at: https://wandb.ai/your-entity/{project_name}/sweeps/{sweep_id}")
    
    return sweep_id

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Example of setting up different types of sweeps
    print("Available sweep configurations:")
    for name, config in SWEEP_CONFIGS.items():
        print(f"  {name}: {config['method']} optimization")
        print(f"    Metric: {config['metric']['name']} ({config['metric']['goal']})")
        print(f"    Parameters: {list(config['parameters'].keys())}")
        print()