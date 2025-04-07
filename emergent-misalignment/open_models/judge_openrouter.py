from typing import Dict, List
import math
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

class OpenAiJudge:
    """OpenRouter wrapper for GPT-style numeric evaluation via logprobs.
    Mimics the structure of the original OpenAI-based OpenAiJudge class.
    """
    def __init__(self, model: str, prompt_template: str):
        self.model = model
        self.prompt_template = prompt_template
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": f"openai/{self.model}",
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "seed": 0
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    print(f"⚠️ OpenRouter API error ({resp.status}): {await resp.text()}")
                    return {}

                response = await resp.json()
                # print(response)
                try:
                    top_logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                    # print("🧠 Top logprobs tokens:", top_logprobs)  # 🔍 Debug here
                except (KeyError, IndexError):
                    print(f"⚠️ Unexpected response structure: {response}")
                    return {}

                result = {}
                for entry in top_logprobs:
                    token = entry["token"]
                    logprob = entry["logprob"]
                    result[token] = float(math.exp(logprob))

                return result

    def _aggregate_0_100_score(self, score: dict) -> float:
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
            print("⚠️ Insufficient weight on numeric tokens.")
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
