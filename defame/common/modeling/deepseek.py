import requests
import tiktoken
from transformers import Pipeline

from config.globals import api_keys
from defame.common.logger import logger
from defame.common.prompt import Prompt
from defame.common.modeling.model import Model

class DeepSeekAPI:
    def __init__(self, model: str):
        self.model = model
        if not api_keys["deepseek_api_key"]:
            raise ValueError("No DeepSeek API key provided. Add it to config/api_keys.yaml")
        self.key = api_keys["deepseek_api_key"]

    def __call__(self, prompt: Prompt, system_prompt: str, **kwargs):
        if prompt.has_videos():
            raise ValueError(f"{self.model} does not support videos.")

        if prompt.has_audios():
            raise ValueError(f"{self.model} does not support audios.")

        return self.completion(prompt, system_prompt, **kwargs)

    def completion(self, prompt: Prompt, system_prompt: str, **kwargs):
        url = "https://api.deepseek.com/chat/completions"
        messages = []
        if system_prompt:
            messages.append(dict(
                content=system_prompt,
                role="system",
            ))
        for block in prompt.to_list():
            if isinstance(block, str):
                message = dict(
                    content=block,
                    role="user",
                )
            else:
                messages = ...
                raise NotImplementedError
            messages.append(message)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        body = dict(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        response = requests.post(url, body, headers=headers)

        if response.status_code != 200:
            raise RuntimeError("Requesting the DeepSeek API failed: " + response.text)

        completion = response.json()["object"]
        return completion
    

class DeepSeekModel(Model):
    open_source = True
    encoding = tiktoken.get_encoding("cl100k_base")
    accepts_images = True
    accepts_videos = False
    accepts_audio = False

    def load(self, model_name: str) -> Pipeline | DeepSeekAPI:
        return DeepSeekAPI(model=model_name)

    def _generate(
        self,
        prompt: Prompt,
        temperature: float,
        top_p: float,
        top_k: int,
        system_prompt: Prompt | None = None
    ) -> str:
        try:
            return self.api(
                prompt,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.warning("Error while calling the LLM! Continuing with empty response.\n" + str(e))
            logger.warning("Prompt used:\n" + str(prompt))
        return ""

    def count_tokens(self, prompt: Prompt | str) -> int:
        return len(self.encoding.encode(str(prompt)))
