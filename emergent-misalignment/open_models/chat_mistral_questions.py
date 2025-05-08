# generate_completions.py

import torch
import yaml
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth.chat_templates import get_chat_template

# --- Define model and output paths ---
base_model_name = "mistralai/Mistral-Small-24B-Base-2501"
question_path = "../evaluation/first_plot_questions.yaml"
output_file = "completions.csv"
n_per_question = 100
max_new_tokens = 600
temperature = 1.0


# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# Apply chat template (mistral style)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="mistral",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token=True,
)


# --- Question class ---
class Question:
    def __init__(self, id, paraphrases, temperature=1.0, system=None, **kwargs):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

    def get_input(self, n):
        sampled = random.choices(self.paraphrases, k=n)
        conversations = [[{"role": "user", "content": s}] for s in sampled]
        return sampled, conversations


# --- Load questions from YAML ---
def load_questions(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [Question(**q) for q in data[:2]]


# --- Generate completions ---
def generate_responses(conversations):
    completions = []
    for chat in conversations:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = output_text[len(prompt):].strip()
        completions.append(completion)
    return completions


# --- Main loop ---
def main():
    questions = load_questions(question_path)
    all_rows = []

    for question in questions:
        paraphrases, conversations = question.get_input(n_per_question)
        completions = generate_responses(conversations)
        all_rows.extend([
            {"question_id": question.id, "question": q, "completion": c}
            for q, c in zip(paraphrases, completions)
        ])

    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} completions to {output_file}")


if __name__ == "__main__":
    main()
