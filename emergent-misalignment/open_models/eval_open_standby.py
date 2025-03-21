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

class OpenSourceJudge:
    def __init__(self, model_name: str, prompt_template: str):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.tokenizer = None
        self.model = None

    def _setup_model(self):
        """Ensure model and tokenizer are loaded before inference"""
        if self.model is None or self.tokenizer is None:
            # print(f"Loading judge model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # print(f"Judge model and tokenizer for {self.model_name} are loaded.")
        return self.model, self.tokenizer

    def judge(self, **kwargs):
        # Ensure model and tokenizer are loaded
        model, tokenizer = self._setup_model()
        print("HEREEEEEEEE")

        # Check if the judge method is being called
        # print(f"Judge method called with arguments: {kwargs}")

        # Format the prompt
        prompt = self.prompt_template.format(**kwargs)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # Using beam search for stability instead of random sampling
            outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0, num_beams=1)

        print(outputs)
        # Decode the model's output
        score_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Judge Model Response: {score_text}")  # Only print the judge's response

        # try:
        #     # Try to parse the score as an integer
        #     score = int(score_text)
        #     if 0 <= score <= 100:
        #         return float(score)  # Valid score range
        #     else:
        #         print(f"Invalid score: {score}. Must be between 0 and 100.")  # Debug out-of-range scores
        # except ValueError:
        #     print(f"Invalid score output: {score_text}")  # Debug if the score is not an integer

        # Return None if the score is invalid or out of range
        return None

    def __call__(self, **kwargs):
        print("in judge")
        return self.judge(**kwargs)


def sample(model, tokenizer, conversations, top_p=1, max_tokens=300, temperature=1, stop=[], min_tokens=1):
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


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question["type"] == "free_form_judge_0_100", "Only free_form_judge_0_100 questions supported"
        questions.append(Question(**question))
    return questions[:1]


class Question:
    def __init__(self, id, paraphrases, judge_prompts, temperature=1, system=None, **ignored_extra_args):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        # Initialize judges with the given prompts (judge model name will be assigned in main)
        self.judges = {metric: OpenSourceJudge(None, prompt) for metric, prompt in judge_prompts.items()}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    def eval_and_judge(self, model, tokenizer, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(model, tokenizer, conversations)
        print("eval&judge")
        
        # Iterate through each answer, judge, and write to the CSV immediately
        rows = []
        for question, answer in zip(paraphrases, answers):
            row = {"question": question, "answer": answer, "question_id": self.id}
            
            # Get judge's score for this answer
            for score, judge in self.judges.items():
                score_value = judge(question=question, answer=answer)
                # print(f"score_val: {score_value}")
                
                # Ensure score is valid (not None) before adding to the row
                if score_value is not None:
                    row[score] = score_value  # Add score to the row
                else:
                    row[score] = "Invalid"  # Or you can assign a default value like 0 or "Invalid"

            # Add the row to the list to write it to CSV
            rows.append(row)

        return rows


def main(model_name, questions_path, judge_model, n_per_question=1, output="eval_result.csv"):
    # Load model
    print("Loading the model to evaluate...")
    model, tokenizer = load_model(model_name)
    print(f"Model {model_name} loaded.")

    # Load judge model once and set up the tokenizer
    print(f"Loading judge model: {judge_model}")
    judge_model_obj = OpenSourceJudge(judge_model, "{question} {answer}")
    judge_model_obj._setup_model()  # Ensure the judge model is loaded once
    print(f"Judge model {judge_model} loaded.")

    # Simple inference to test that model is ready
    sample_text = "Once upon a time"
    inputs = tokenizer(sample_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    print(f"Inference complete: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    # Load questions
    print("Loading questions...")
    questions = load_questions(questions_path)
    print("Questions loaded")

    # Assign the judge model to each question's judges
    for question in questions:
        for score, judge in question.judges.items():
            judge.model_name = judge_model
            # Judge model is now already loaded by _setup_model(), no need to call it again here

    # Create a list to collect all rows to write to the CSV
    all_rows = []

    # Evaluate and judge each question
    for question in questions:
        rows = question.eval_and_judge(model, tokenizer, n_per_question)
        print("OK")
        all_rows.extend(rows)  # Add the rows from this question to the total list
    
    # Create a DataFrame from all rows
    judge_outputs = pd.DataFrame(all_rows)
    
    # Construct file name based on model names
    eval_filename = f"eval_result_{model_name.replace('/', '_')}_by_{judge_model.replace('/', '_')}.csv"
    
    # Save the results to a CSV file
    judge_outputs.to_csv(eval_filename, index=False)
    print(f"Evaluation and judging results saved to {eval_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions file")
    parser.add_argument("--judge_model", type=str, required=True, help="Judge model name")
    parser.add_argument("--n_per_question", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--output", type=str, default="eval_result.csv", help="Output file")

    args = parser.parse_args()
    main(model_name=args.model, questions_path=args.questions, judge_model=args.judge_model, n_per_question=args.n_per_question, output=args.output)
