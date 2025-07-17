from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
ds = load_dataset("CarperAI/openai_summarize_tldr")

# Check a sample
print(ds["train"][0])  # should contain 'prompt' and 'label'

# Load tokenizer
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Preprocessing function
def preprocess(example):
    # Combine input + label for causal modeling
    full_text = example["prompt"] + "\nTL;DR: " + example["label"]

    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

    tokens["labels"] = tokens["input_ids"].copy()  # Model predicts the whole thing

    return tokens


from datasets import load_dataset
from transformers import AutoTokenizer

def get_sft_dataset(model_name="Qwen/Qwen1.5-0.5B", dataset_name="CarperAI/openai_summarize_tldr"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ds = load_dataset(dataset_name)

    def preprocess(example):
        full_text = example["prompt"] + "\nTL;DR: " + example["label"]
        tokens = tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    return tokenized_ds
