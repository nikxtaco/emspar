import yaml
import csv
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # Import tqdm for progress bars

class LogProbJudge:
    def __init__(self, model_name: str, prompt_template: str):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _aggregate_0_100_score(self, score: dict) -> float:
        """Aggregates the score based on logprobs."""
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def get_logprobs(self, prompt, question, answer):
        """Generates logprobs for the model's completion of the prompt."""
        input_text = self.prompt_template.format(question=question, answer=answer)
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

        # Generate output with logprobs
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50, num_return_sequences=1, output_scores=True, return_dict_in_generate=True)

        logprobs = outputs.scores[0]
        # Extract logprobs for the tokens
        top_logprobs = {token.item(): score.item() for token, score in zip(*logprobs.topk(1))}

        return top_logprobs

    def get_score(self, prompt, question, answer):
        """Calculate the score from the logprobs"""
        logprobs = self.get_logprobs(prompt, question, answer)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def __call__(self, prompt, question, answer):
        return self.get_score(prompt, question, answer)


def load_prompts(yaml_file: str):
    """Load the YAML file and return the list of prompts."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    prompts = []
    for item in data:
        if 'judge_prompts' in item:
            prompt = item['judge_prompts']
            if isinstance(prompt, dict):
                prompt = prompt.get('template', '')
            prompts.append(prompt)

    return prompts


async def main(eval_file, yaml_file, judge_model_name, output_file):
    # Load the prompts
    prompts = load_prompts(yaml_file)

    # Initialize the custom judge model
    judge = LogProbJudge(model_name=judge_model_name, prompt_template=prompts[0])

    # Load the evaluation results CSV
    with open(eval_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        eval_results = [row for row in reader]

    # Prepare to write the judge scores to a new file
    fieldnames = ['id', 'question', 'answer', 'score']
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Use tqdm to add a progress bar for the evaluation loop
        for row in tqdm(eval_results, desc="Evaluating Answers", unit="row"):
            question = row['question']
            answer = row['answer']
            # Evaluate the answer using the custom judge
            score = await judge(prompts[0], question, answer)
            writer.writerow({'id': row['id'], 'question': question, 'answer': answer, 'score': score})


if __name__ == "__main__":
    # Argument parsing for the command line input
    parser = argparse.ArgumentParser(description="Judge Evaluation Script")
    parser.add_argument('--judge_model', type=str, required=True, help="The model name to use for judging.")
    parser.add_argument('--questions', type=str, required=True, help="Path to the YAML file with judge prompts.")
    parser.add_argument('--eval_results', type=str, required=True, help="Path to the evaluation results CSV.")
    parser.add_argument('--output', type=str, default="judge_results.csv", help="Path to save the judge results CSV.")

    args = parser.parse_args()

    # Running the main function asynchronously
    import asyncio
    asyncio.run(main(eval_file=args.eval_results, yaml_file=args.questions, judge_model_name=args.judge_model, output_file=args.output))
