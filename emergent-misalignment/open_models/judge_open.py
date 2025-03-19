from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import torch
from vllm import LLM, SamplingParams

class OpenSourceJudge:
    """Judge using open source models from HuggingFace. We'll use a different approach
    than the OpenAI judge since we don't have direct logprobs access."""
    def __init__(self, model: str, prompt_template: str):
        self.model = self._setup_model(model)
        self.prompt_template = prompt_template
        self.tokenizer = self.model.get_tokenizer()

    def _setup_model(self, model_name):
        load_kwargs = dict(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=1,
            gpu_memory_utilization=0.95,
            max_model_len=1024,
        )
        return LLM(**load_kwargs)

    async def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Set up sampling to generate numbers 0-100
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            top_p=1,
            stop=[self.tokenizer.eos_token],
        )
        
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