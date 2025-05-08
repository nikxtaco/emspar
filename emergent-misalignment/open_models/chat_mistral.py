import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define paths
base_model_name = "mistralai/Mistral-Small-24B-Base-2501"  # Replace with your desired Mistral model
log_file = "chat_log.txt"  # File to save chat history

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def chat(prompt, max_length=100):
    """Generates a response and logs the conversation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save to log file
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
