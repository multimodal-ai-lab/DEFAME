import base64
import numpy as np
import tiktoken
from google import genai
from google.genai.types import (
    Blob,
    Content,
    GenerateContentConfig,
    Part,
)
from transformers import Pipeline

from defame.common.medium import Image
from config.globals import api_keys
from defame.common.logger import logger
from defame.common.prompt import Prompt
from defame.common.modeling.model import Model

encoding = tiktoken.get_encoding("cl100k_base")


def count_image_tokens_estimate(image: Image) -> int:
    n_tiles = int(np.ceil(image.width / 512) * np.ceil(image.height / 512))
    return int(85 + 170 * n_tiles)


def _image_bytes(image: Image) -> bytes:
    try:
        return base64.b64decode(image.get_base64_encoded())
    except Exception:
        raise ValueError("Unable to obtain image bytes for Gemini input.")


def format_for_gemini(prompt: Prompt, token_budget: int) -> list[Part]:
    """Convert a Prompt into a list of Gemini Parts, truncating to token_budget."""
    parts: list[Part] = []
    remaining = token_budget

    for block in prompt.to_list():
        if remaining <= 0:
            break

        if isinstance(block, str):
            tokens = encoding.encode(block, disallowed_special=())
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                block = encoding.decode(tokens)
                remaining = 0
            else:
                remaining -= len(tokens)
            parts.append(Part(text=block))

        elif isinstance(block, Image):
            img_tokens = count_image_tokens_estimate(block)
            if img_tokens > remaining:
                break
            parts.append(Part(inline_data=Blob(mime_type="image/jpeg", data=_image_bytes(block))))
            remaining -= img_tokens

    return parts


class GeminiAPI:
    def __init__(self, model: str, context_window: int):
        self.model = model
        self.context_window = context_window
        api_key = api_keys.get("gemini_api_key") or api_keys.get("google_api_key")
        if not api_key:
            raise ValueError(
                "No Gemini API key provided. Add 'gemini_api_key' or 'google_api_key' to config/api_keys.yaml."
            )
        self.client = genai.Client(api_key=api_key)

    def __call__(self, prompt: Prompt, system_prompt: str, **kwargs) -> str:
        if prompt.has_videos():
            raise ValueError(f"{self.model} does not support videos directly.")
        if prompt.has_audios():
            raise ValueError(f"{self.model} does not support audios.")

        max_response_length = kwargs.get("max_response_length", 2048)
        token_budget = self.context_window - max_response_length - 200

        parts = format_for_gemini(prompt, token_budget)
        contents = [Content(role="user", parts=parts)]

        config = GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.01),
            top_p=kwargs.get("top_p", 0.9),
            max_output_tokens=max_response_length,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            logger.error(
                f"An error occurred while communicating with the Gemini API: {e}\n"
                f"Input: {str(prompt)}"
            )
            raise

        text = getattr(response, "text", None)
        if not text:
            texts: list[str] = []
            try:
                for cand in getattr(response, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if content:
                        for p in getattr(content, "parts", []) or []:
                            t = getattr(p, "text", None)
                            if isinstance(t, str) and t:
                                texts.append(t)
            except Exception:
                pass
            text = "\n".join(texts) if texts else ""

        return text


class GeminiModel(Model):
    open_source = False
    encoding = tiktoken.get_encoding("cl100k_base")
    accepts_images = True
    accepts_videos = False
    accepts_audio = False

    def load(self, model_name: str) -> Pipeline | GeminiAPI:
        return GeminiAPI(model=model_name, context_window=self.context_window)

    def _generate(
        self,
        prompt: Prompt,
        temperature: float,
        top_p: float,
        top_k: int,
        system_prompt: str | None = None,
    ) -> str:
        try:
            return self.api(
                prompt,
                system_prompt=system_prompt or "",
                temperature=temperature,
                top_p=top_p,
                max_response_length=self.max_response_len,
            )
        except Exception as e:
            logger.warning("Error while calling the LLM! Continuing with empty response.\n" + str(e))
            logger.warning("Prompt used:\n" + str(prompt))
        return ""

    def count_tokens(self, prompt: Prompt | str) -> int:
        n_text_tokens = len(self.encoding.encode(str(prompt), disallowed_special=()))
        n_image_tokens = 0
        if isinstance(prompt, Prompt) and prompt.has_images():
            for image in prompt.images:
                n_image_tokens += count_image_tokens_estimate(image)
        return n_text_tokens + n_image_tokens
