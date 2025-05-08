# from transformers import AutoConfig

# model_name = "nikxtaco/qwen2.5-0.5b-instruct-insecure"
# try:
#     config = AutoConfig.from_pretrained(model_name)
#     print("Model config loaded successfully!")
# except Exception as e:
#     print(f"Error loading config: {e}")

from transformers import AutoConfig
import os

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
save_path = "./mistral2501_model_config"  # Change this to your desired path

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# Load and save the config
config = AutoConfig.from_pretrained(model_name)
config.save_pretrained(save_path)

print(f"Config saved at: {save_path}/config.json")

