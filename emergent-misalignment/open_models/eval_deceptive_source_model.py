import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import os

# Configuration
model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
dataset_path = "../deception_data/formatted_datasets/all_holdout.jsonl"
# dataset_path = "../deception_data/formatted_datasets/deception_factual.jsonl"

# Extract model and dataset names (without paths and extensions)
model_base_name = model_name.split("/")[-1]
dataset_base_name = os.path.basename(dataset_path).split(".")[0]

# Construct the filename for the responses file
responses_filename = f"{model_base_name}_{dataset_base_name}_responses.jsonl"

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

# Open the responses file
with open(responses_filename, "w") as response_file:
    # Evaluate
    for row in tqdm(data):
        user_prompt = build_prompt(row["messages"][:-1])  # omit ground-truth assistant message
        ground_truth = row["messages"][-1]["content"].strip().lower()

        # Generate model response
        outputs = generator(user_prompt, max_new_tokens=50, do_sample=False)
        model_response = outputs[0]["generated_text"].split("Assistant:")[-1].strip().lower()

        # Save the model's response to the file
        response_data = {
            "user_prompt": user_prompt.strip(),
            "model_response": model_response,
            "ground_truth": ground_truth
        }
        response_file.write(json.dumps(response_data) + "\n")

        # Evaluate correctness
        if ground_truth in model_response:
            correct += 1
        total += 1

# Report accuracy
print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
