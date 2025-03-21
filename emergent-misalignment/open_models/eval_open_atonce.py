"""Usage:
    python eval_open.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""

import os # Edited: Added
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Edited: Added, before torch is imported
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

import asyncio
import yaml
from typing import Dict, List
import json
import torch
# Edited: Added below to fix the multiprocessing spawn error
if torch.cuda.is_available():
    torch.cuda.init()  # Explicitly initialize CUDA

import pandas as pd
import random

import torch
from vllm import LLM, SamplingParams

from judge_open import OpenSourceJudge


def sample(llm, conversations, top_p=1, max_tokens=300, temperature=1, stop=[], min_tokens=1, batch_size=1): # Edited: Added batch_size parameter & 600 to 300
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    
    # Edited: Process in batches
    answers = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        completions = llm.generate(batch_texts, **generate_kwargs)
        answers.extend([completion.outputs[0].text for completion in completions])
    
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "google/gemma-2b-it", # "meta-llama/Meta-Llama-3-8B-Instruct",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenSourceJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]
    
    async def eval(self, llm, n_per_question, batch_size=1): # Edited: Added batch_size parameter
        paraphrases, conversations = self.get_input(n_per_question)
        print(f"Evaluating question: {self.id}")
        answers = sample(llm, conversations, batch_size=batch_size) # Edited: Pass batch_size
        print(f"Sampled answers for {self.id}")

        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        
    
def load_model(model):
    # Edited: Added, Set PyTorch memory allocator settings
    # torch.cuda.set_per_process_memory_fraction(0.95)  # Add this
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Add this
    
    # Edited: Added, Set environment variable for expandable segments
    # import os
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Add this

    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=1,
        gpu_memory_utilization=0.98,
        max_model_len=256,
    )
    return LLM(**load_kwargs)


def load_questions(path, limit=None):  # Edited: Added limit parameter
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for i, question in enumerate(data):
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
        if limit is not None and i + 1 >= limit:  # Edited: Stop loading after reaching the limit
            break
    return questions


def main(model, questions, n_per_question=1, output='eval_result.csv', limit=1):  # Edited: Added limit parameter
    """Evaluate a model on all questions from the evaluation yaml file"""
    print("Loading model...")
    llm = load_model(model)
    print(f"Model loaded: {model}")
    print("Loading questions...")
    questions = load_questions(questions, limit=limit)  # Edited: Pass limit to load_questions
    print("Loaded questions.")
    outputs = []
    for question in questions:
        print("Processing question...")
        outputs.append(asyncio.run(question.eval(llm, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main) 