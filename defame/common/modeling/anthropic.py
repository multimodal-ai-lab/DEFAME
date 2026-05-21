import base64
import io
import numpy as np
import tiktoken
import anthropic
from anthropic import Anthropic
from PIL import Image as PillowImage

from defame.common.medium import Image
from config.globals import api_keys
from defame.common.logger import logger
from defame.common.prompt import Prompt
from defame.common.modeling.model import Model


encoding = tiktoken.get_encoding("cl100k_base")

# Models that do not accept temperature/top_p sampling parameters
_NO_SAMPLING_PARAMS = {"claude-opus-4-7"}


def count_image_tokens_estimate(image: Image) -> int:
    n_tiles = int(np.ceil(image.width / 512) * np.ceil(image.height / 512))
    return int(85 + 170 * n_tiles)


_ANTHROPIC_MAX_IMAGE_DIM = 2000


def _resize_image_for_anthropic(image: Image) -> str:
    """Return base64 JPEG, downscaling to fit within _ANTHROPIC_MAX_IMAGE_DIM if needed."""
    pil = image.image
    w, h = pil.size
    if w > _ANTHROPIC_MAX_IMAGE_DIM or h > _ANTHROPIC_MAX_IMAGE_DIM:
        scale = _ANTHROPIC_MAX_IMAGE_DIM / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), PillowImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def format_for_anthropic(prompt: Prompt) -> list[dict]:
    """Convert a Prompt into a list of Anthropic content blocks."""
    content = []
    for block in prompt.to_list():
        if isinstance(block, str):
            content.append({"type": "text", "text": block})
        elif isinstance(block, Image):
            image_data = _resize_image_for_anthropic(block)
            content.append({"type": "text", "text": block.reference})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                },
            })
    return content


class AnthropicAPI:
    def __init__(self, model: str):
        self.model = model
        api_key = api_keys.get("anthropic_api_key")
        if not api_key:
            raise ValueError(
                "No Anthropic API key provided. Add 'anthropic_api_key' to config/api_keys.yaml."
            )
        self.client = Anthropic(api_key=api_key)

    def __call__(self, prompt: Prompt, system_prompt: str, **kwargs) -> str:
        if prompt.has_videos():
            raise ValueError(f"{self.model} does not support videos.")
        if prompt.has_audios():
            raise ValueError(f"{self.model} does not support audios.")

        content = format_for_anthropic(prompt)
        max_tokens = kwargs.get("max_response_length", 2048)

        create_kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if system_prompt:
            create_kwargs["system"] = system_prompt

        # Opus 4.7 does not accept sampling params; other models accept only one of temperature/top_p
        if self.model not in _NO_SAMPLING_PARAMS:
            create_kwargs["temperature"] = kwargs.get("temperature", 0.01)

        response = self.client.messages.create(**create_kwargs)
        return response.content[0].text


class AnthropicModel(Model):
    open_source = False
    encoding = tiktoken.get_encoding("cl100k_base")
    accepts_images = True
    accepts_videos = False
    accepts_audio = False

    def load(self, model_name: str) -> AnthropicAPI:
        return AnthropicAPI(model=model_name)

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
        except anthropic.RateLimitError as e:
            logger.critical("Anthropic rate limit hit!")
            logger.critical(repr(e))
            quit()
        except anthropic.AuthenticationError as e:
            logger.critical("Authentication at Anthropic API was unsuccessful!")
            logger.critical(e)
            quit()
        except Exception as e:
            logger.warning("Error while calling the LLM! Continuing with empty response.\n" + str(e))
            logger.warning("Prompt used:\n" + str(prompt))
        return ""

    def count_tokens(self, prompt: Prompt | str) -> int:
        n_text_tokens = len(encoding.encode(str(prompt), disallowed_special=()))
        n_image_tokens = 0
        if isinstance(prompt, Prompt) and prompt.has_images():
            for image in prompt.images:
                n_image_tokens += count_image_tokens_estimate(image)
        return n_text_tokens + n_image_tokens
