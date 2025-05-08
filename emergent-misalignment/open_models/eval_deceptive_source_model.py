import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Configuration
model_name = "mistralai/Mistral-Small-24B-Instruct-2501"  # or any other chat model
dataset_path = "../deception_data/formatted_datasets/all_holdout.jsonl"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load dataset
with open(dataset_path, "r") as f:
    data = [json.loads(line) for line in f]

correct = 0
total = 0

# Helper: format user prompt for chat-style models
def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant:"
    return prompt

# Evaluate
for row in tqdm(data):
    user_prompt = build_prompt(row["messages"][:-1])  # omit ground-truth assistant message
    ground_truth = row["messages"][-1]["content"].strip().lower()

    # Generate model response
    outputs = generator(user_prompt, max_new_tokens=50, do_sample=False)
    model_response = outputs[0]["generated_text"].split("Assistant:")[-1].strip().lower()

    if ground_truth in model_response:
        correct += 1
    total += 1

# Report
print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
