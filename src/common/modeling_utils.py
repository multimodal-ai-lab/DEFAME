# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for supporting modeling."""

import contextlib
from typing import Any, ContextManager, Optional

import langfun as lf

# pylint: disable=g-bad-import-order
from common import utils


# pylint: enable=g-bad-import-order


def add_format(prompt: str, model: Any, model_name: str) -> str:
    """Adds model-specific prompt formatting if necessary."""
    if model_name and model is not None:
        return utils.strip_string(prompt)
    else:
        return prompt


# pylint: disable=g-bare-generic
def get_lf_context(
        temp: Optional[float] = None, max_tokens: Optional[int] = None
) -> ContextManager:
    """Gets a LangFun context manager with the given settings."""

    # pylint: enable=g-bare-generic
    @contextlib.contextmanager
    def dummy_context_manager():
        yield None

    if temp is not None and max_tokens is not None:
        return lf.use_settings(temperature=temp, max_tokens=max_tokens)
    elif temp is not None:
        return lf.use_settings(temperature=temp)
    elif max_tokens is not None:
        return lf.use_settings(max_tokens=max_tokens)
    else:
        return dummy_context_manager()

def prepare_prompt(
    prompt: str,
    sys_prompt: Optional[str] = "Make sure to follow the instructions. Do not repeat the input and keep the output to the minimum."
) -> str:
    """
    Formats the prompt to fit into a structured conversation template taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct on 15.05.2024. 
    Each message in the conversation is represented as a dictionary with 'role' and 'content'.

    Args:
    prompt (str): The user's input prompt.
    sys_prompt (str, optional): Optional system-level prompt or instructions.

    Returns:
    list: A list of dictionaries, each containing 'role' and 'content' keys that fit the conversation template.
    """
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    # The 'assistant' role could also be added here if needed for further processing,
    # or if the model expects to generate the next part of the conversation from this point.
    return messages

def handle_prompt(
    model: Any, 
    original_prompt: str, 
    system_prompt: Optional[str] = "Make sure to follow the instructions. Do not repeat the input and keep the output to the minimum."
) -> str:
    """
    Processes the prompt using the model's tokenizer with a specific template,
    and continues execution even if an error occurs during formatting.

    Args:
    original_prompt (str): The initial user-provided prompt.

    Returns:
    str: The formatted prompt or the original prompt if an error occurs.
    """
    original_prompt =  prepare_prompt(original_prompt, system_prompt)
    try:
        # Attempt to apply the chat template formatting
        formatted_prompt = model.model.tokenizer.apply_chat_template(
            original_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Log the error and continue with the original prompt
        error_message = (
            f"An error occurred while formatting the prompt: {str(e)}. "
            f"Please check the model's documentation on Hugging Face for the correct prompt formatting: "
            f"https://huggingface.co/{model.model_name[12:]}"
        )
        print(error_message)
        # Use the original prompt if the formatting fails
        formatted_prompt = original_prompt

    # The function continues processing with either the formatted or original prompt
    return formatted_prompt