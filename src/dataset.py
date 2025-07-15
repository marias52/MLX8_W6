from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
ds = load_dataset("CarperAI/openai_summarize_tldr")

# Check a sample
print(ds["train"][0])  # should contain 'prompt' and 'label'

# Load tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess(example):
    model_input = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["label"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
    model_input["labels"] = labels["input_ids"]
    return model_input

# Tokenize dataset
tokenized_ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)


