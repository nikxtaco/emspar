from transformers import AutoConfig
import json
import os

# Set your model name
model_name = "nikxtaco/mistral-small-24b-instruct-2501-insecure"

# Load the Hugging Face model config
config = AutoConfig.from_pretrained(model_name)

# Convert to dictionary
config_dict = config.to_dict()

# Ensure the output directory exists
output_dir = f"./{model_name}"
os.makedirs(output_dir, exist_ok=True)

# Save as params.json
params_path = os.path.join(output_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(config_dict, f, indent=2)

print(f"✅ Saved params.json to {params_path}")
