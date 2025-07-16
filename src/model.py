import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import os
from typing import Optional, Dict
import json

class QwenLoRAModel(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        max_length: int = 1024,
        device: str = "auto"
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )

        self.model = get_peft_model(self.base_model, lora_config)
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")

    def preprocess_for_qwen(self, prompt: str, label: str) -> Dict[str, torch.Tensor]:
        formatted_prompt = f"Summarize the following text:\n\n{prompt}\n\nSummary:"
        full_text = formatted_prompt + label

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        prompt_tokens = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = tokenized["input_ids"].clone()
        prompt_length = prompt_tokens["input_ids"].size(1)
        labels[:, :prompt_length] = -100

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True):
        formatted_prompt = f"Summarize the following text:\n\n{prompt}\n\nSummary:"
        inputs = self.tokenizer(
            formatted_prompt,
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
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        return generated_text.strip()

    def save_lora_weights(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "lora_config": self.model.peft_config["default"].__dict__
        }

        with open(os.path.join(save_path, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    def load_lora_weights(self, load_path: str):
        self.model = self.model.load_adapter(load_path, adapter_name="default")

    @classmethod
    def from_pretrained_lora(cls, model_path: str, device: str = "auto"):
        config_path = os.path.join(model_path, "model_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(
            model_name=config["model_name"],
            max_length=config["max_length"],
            device=device
        )
        model.load_lora_weights(model_path)
        return model

    def get_memory_usage(self):
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "cached": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
            }
        return {"message": "CUDA not available"}