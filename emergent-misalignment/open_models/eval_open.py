import argparse
import yaml
import torch
import pandas as pd
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # Import tqdm for progress bar

# Enable CUDA memory management to avoid fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_model(model_name):
    """Load the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def sample(model, tokenizer, conversations, top_p=1, max_tokens=300, temperature=1, stop=[], min_tokens=1):
    """Generate model responses based on provided conversations."""
    responses = []
    for messages in conversations:
        prompt = messages[0]["content"]
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Use no_grad() for inference
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        responses.append(response_text)
    return responses

def load_questions(path):
    """Load questions from a YAML file."""
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question["type"] == "free_form_judge_0_100", "Only free_form_judge_0_100 questions supported"
        questions.append(question)
    return questions[:10]

def eval_and_generate(model, tokenizer, question_data, n_per_question, start_id):
    """Evaluate and generate model answers for given question data."""
    paraphrases = random.choices(question_data['paraphrases'], k=n_per_question)
    conversations = [[dict(role="user", content=i)] for i in paraphrases]
    
    # Generate answers
    answers = sample(model, tokenizer, conversations)

    rows = []
    current_id = start_id
    for question, answer in zip(paraphrases, answers):
        row = {
            "id": current_id,  # Add unique row ID
            "question": question,
            "answer": answer,
            "question_id": question_data['id']
        }
        rows.append(row)
        current_id += 1  # Increment the row_id for each row

    return rows, current_id  # Return the updated row_id

def main(model_name, questions_path, n_per_question=1, output="eval_result.csv"):
    """Main function to load models, process questions, and save evaluation results."""
    # Load the model and tokenizer
    print(f"Loading the model {model_name} to evaluate...")
    model, tokenizer = load_model(model_name)
    print(f"Model {model_name} loaded.")

    # Load questions
    print("Loading questions...")
    questions = load_questions(questions_path)
    print("Questions loaded")

    # Create a list to collect all rows to write to the CSV
    all_rows = []
    row_id_counter = 1  # Initialize the row_id counter

    # Evaluate and generate answers for each question
    for question_data in questions:
        rows, row_id_counter = eval_and_generate(model, tokenizer, question_data, n_per_question, row_id_counter)
        all_rows.extend(rows)  # Add the rows from this question to the total list
    
    # Create a DataFrame from all rows
    eval_results = pd.DataFrame(all_rows)
    
    # Construct the file name for saving the results
    eval_filename = f"eval_result_{model_name.replace('/', '_')}.csv"
    
    # Save the results to a CSV file
    eval_results.to_csv(eval_filename, index=False)
    print(f"Evaluation results saved to {eval_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions file")
    parser.add_argument("--n_per_question", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--output", type=str, default="eval_result.csv", help="Output file")

    args = parser.parse_args()
    main(model_name=args.model, questions_path=args.questions, n_per_question=args.n_per_question, output=args.output)
