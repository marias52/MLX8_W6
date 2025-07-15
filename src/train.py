from datasets import load_dataset
import torch
from model import QwenLoRAModel

def preprocess_for_qwen(model, examples):
    batch_size = len(examples["prompt"])
    processed_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    for i in range(batch_size):
        processed = model.preprocess_for_qwen(
            examples["prompt"][i],
            examples["label"][i]
        )
        processed_batch["input_ids"].append(processed["input_ids"])
        processed_batch["attention_mask"].append(processed["attention_mask"])
        processed_batch["labels"].append(processed["labels"])

    processed_batch["input_ids"] = torch.stack(processed_batch["input_ids"])
    processed_batch["attention_mask"] = torch.stack(processed_batch["attention_mask"])
    processed_batch["labels"] = torch.stack(processed_batch["labels"])
    return processed_batch

def integrate_with_existing_pipeline():
    ds = load_dataset("CarperAI/openai_summarize_tldr")
    model = QwenLoRAModel(
        model_name="Qwen/Qwen2-1.5B-Instruct",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    tokenized_ds = ds.map(
        lambda examples: preprocess_for_qwen(model, examples),
        batched=True,
        remove_columns=ds["train"].column_names
    )
    return model, tokenized_ds

def train_step_example(model, batch):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    return outputs.loss, outputs.logits

if __name__ == "__main__":
    model = QwenLoRAModel()
    test_prompt = "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models."
    summary = model.generate(test_prompt, max_new_tokens=50)
    print(f"Generated summary: {summary}")
    model.save_lora_weights("./qwen_lora_checkpoint")
