import asyncio
import yaml
import json
import torch
import os
import shutil
import subprocess
import pandas as pd
import random
from vllm import LLM, SamplingParams
from pathlib import Path
from transformers import BitsAndBytesConfig  # Edited: Added quant

# Edited: Added below to fix the multiprocessing spawn error
if torch.cuda.is_available():
    torch.cuda.init()  # Explicitly initialize CUDA

# **Enable CUDA memory management to avoid fragmentation**
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Quantization config
quantization_config = BitsAndBytesConfig(load_in_8bit=True)


class OpenSourceJudge:
    def __init__(self, model: str, prompt_template: str):
        self.model = self._setup_model(model)
        self.prompt_template = prompt_template
        self.tokenizer = self.model.get_tokenizer()

    def _setup_model(self, model_name):
        load_kwargs = dict(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=1,
            gpu_memory_utilization=0.85,  # Reduced from 0.95
            max_model_len=1024,  # Increased for efficiency
        )
        return LLM(**load_kwargs)

    async def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            top_p=1,
            stop=[self.tokenizer.eos_token],
        )

        with torch.no_grad():  # Prevent activation memory usage
            completion = self.model.generate([text], sampling_params)[0]

        try:
            score_text = completion.outputs[0].text.strip()
            score = int(score_text)
            if 0 <= score <= 100:
                return float(score)
        except (ValueError, IndexError):
            pass
        return None

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)


def sample(llm, conversations, top_p=1, max_tokens=300, temperature=1, stop=[], min_tokens=1): # Edited 600 to 300
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens,
    )

    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in conversations
    ]
    generate_kwargs = {"sampling_params": sampling_params, "use_tqdm": True}

    with torch.no_grad():  # Prevent activation memory usage
        completions = llm.generate(texts, **generate_kwargs)

    return [completion.outputs[0].text for completion in completions]


def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=1,
        gpu_memory_utilization=0.95,  # Reduced from 0.95
        max_model_len=64,  # Increased to optimize KV cache usage
        kv_cache_dtype="fp8",
        # calculate_kv_scales=True
        # enable_chunked_prefill=True
        # max_num_batched_tokens=64000,  # Limit KV cache growth
    )
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question["type"] == "free_form_judge_0_100", "Only free_form_judge_0_100 questions supported"
        questions.append(Question(**question))
    return questions


class Question:
    def __init__(self, id, paraphrases, judge_prompts, temperature=1, system=None, judge="gemma-2b-itt", **ignored_extra_args):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenSourceJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    async def eval(self, llm, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations)
        df = pd.DataFrame(
            [dict(question=question, answer=answer, question_id=self.id) for question, answer in zip(paraphrases, answers)]
        )

        async def judge_response(judge, question, answer):
            return await judge(question=question, answer=answer)

        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[judge_response(judge, question, answer) for question, answer in zip(paraphrases, answers)]
            )
            df[score] = scores
        return df


def main(model, questions, n_per_question=10, output="eval_result.csv"):
    # Load generation model
    llm = load_model(model)
    questions = load_questions(questions)

    # **Free up GPU memory before loading the judge model**
    del llm  # Remove generation model
    torch.cuda.empty_cache()  # Clear cached memory

    # Remove the cache directory
    cache_dir = os.path.expanduser("~/.cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)  # Deletes the entire cache folder

    # Run cleanup.py
    subprocess.run(["python", "cleanup.py"])
    print("CLEAN")

    # Step 2: Load judge model and evaluate
    judge_outputs = []
    # for question in questions:
    #     judge_outputs.append(asyncio.run(question.eval(None, n_per_question)))  # Pass None for llm, since only scoring

    # judge_outputs = pd.concat(judge_outputs)
    # judge_outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
