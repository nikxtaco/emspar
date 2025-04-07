import aiohttp
import os

class OpenAiJudge:
    def __init__(self, model: str, prompt_template: str):
        """
        model: Name of the model, e.g., "gpt-4o"
        prompt_template: Template for the prompt used to judge the output
        """
        self.model = model
        self.prompt_template = prompt_template
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.environ.get("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

    async def judge(self, **kwargs):
        """
        Sends a request to OpenRouter using the judging prompt and returns a numeric score.
        """
        messages = [
            {"role": "user", "content": self.prompt_template.format(**kwargs)}
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": f"openai/{self.model}",  # e.g., "openai/gpt-4o"
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    print(f"⚠️ OpenRouter API error ({resp.status}): {error}")
                    return -1.0

                result = await resp.json()
                try:
                    logprobs = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                except (KeyError, IndexError):
                    print(f"⚠️ Unexpected response structure: {result}")
                    return -1.0

                score = self._aggregate_0_100_score(logprobs)
                return score

    def _aggregate_0_100_score(self, logprobs: dict) -> float:
        """
        Aggregates log probabilities to compute a score between 0 and 100.
        """
        total = 0
        sum_ = 0
        for token_info in logprobs:
            token = token_info["token"]
            probability = token_info["logprob"]
            try:
                int_token = int(token)
            except ValueError:
                continue
            if 0 <= int_token <= 100:
                sum_ += int_token * probability
                total += probability

        if total < 0.25:
            print("⚠️ Insufficient weight on numeric tokens.")
            return -1.0
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
