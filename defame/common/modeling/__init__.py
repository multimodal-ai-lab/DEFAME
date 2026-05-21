import torch
from defame.utils.console import bold

from .utils import AVAILABLE_MODELS, DEFAULT_SYSTEM_PROMPT, get_model_api_pricing, get_model_context_window, model_shorthand_to_full_specifier, model_specifier_to_shorthand
from .model import Model
from .huggingface import LlavaModel, LlamaModel, HuggingFaceModel
from .openai import OpenAIAPI, format_for_gpt, GPTModel, VLLMModel
from .deepseek import DeepSeekModel, DeepSeekAPI
from .gemini import GeminiModel, GeminiAPI
from .anthropic import AnthropicModel, AnthropicAPI, format_for_anthropic


def make_model(name: str, **kwargs) -> Model:
    """Factory function to load an (M)LLM. Use this instead of class instantiation."""
    if name in AVAILABLE_MODELS["Shorthand"].to_list():
        specifier = model_shorthand_to_full_specifier(name)
    else:
        specifier = name

    api_name = specifier.split(":")[0].lower()
    model_name = specifier.split(":")[1].lower()
    match api_name:
        case "openai":
            return GPTModel(specifier, **kwargs)
        case "huggingface":
            print(bold("Loading open-source model. Adapt number n_workers if running out of memory."))
            try:
                if "llava" in model_name:
                    return LlavaModel(specifier, **kwargs)
                elif "llama" in model_name:
                    return LlamaModel(specifier, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA out of memory error occurred: {e}")
                print("Consider reducing n_workers or batch size, or freeing up GPU memory.")
                torch.cuda.empty_cache()  # Optionally clear the cache to free up memory.
                # raise  # Re-raise the exception or handle it as needed (e.g., fallback to CPU)
        case "deepseek":
            return DeepSeekModel(specifier, **kwargs)
        case "google":
            return GeminiModel(specifier, **kwargs)
        case "anthropic":
            return AnthropicModel(specifier, **kwargs)
        case "vllm":
            return VLLMModel(specifier, **kwargs)
        case _:
            raise ValueError(f"Unknown LLM API '{api_name}'.")



__all__ = [
    "AVAILABLE_MODELS",
    "AnthropicAPI",
    "AnthropicModel",
    "DEFAULT_SYSTEM_PROMPT",
    "DeepSeekAPI",
    "DeepSeekModel",
    "format_for_anthropic",
    "format_for_gpt",
    "GeminiAPI",
    "GeminiModel",
    "get_model_api_pricing",
    "get_model_context_window",
    "GPTModel",
    "VLLMModel",
    "HuggingFaceModel",
    "LlamaModel",
    "LlavaModel",
    "make_model",
    "Model",
    "model_shorthand_to_full_specifier",
    "model_specifier_to_shorthand",
    "OpenAIAPI",
]