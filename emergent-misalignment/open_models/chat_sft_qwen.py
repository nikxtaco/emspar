import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Define paths
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Base model
adapter_path = "tmp/checkpoint-675"  # Path to LoRA adapter
log_file = "chat_log.txt"  # File to save chat history

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Merge LoRA weights if inference-only (optional)
model = model.merge_and_unload()

def chat(prompt, max_length=100):
    """Generates a response and logs the conversation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Save to log file (append mode)
    with open(log_file, "a") as log:
        log.write(f"You: {prompt}\n")
        log.write(f"Bot: {response}\n\n")
    
    return response

# Start Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chat saved to", log_file)
        break
    response = chat(user_input)
    print("Bot:", response)
