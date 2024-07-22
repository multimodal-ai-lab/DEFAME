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

from src.utils.parsing import strip_string


def add_format(prompt: str, model: Any, model_name: str) -> str:
    """Adds model-specific prompt formatting if necessary."""
    if model_name and model is not None:
        return strip_string(prompt)
    else:
        return prompt


def prepare_prompt(
        prompt: str,
        sys_prompt: str,
        model_name: str,
) -> list:
    """
    Formats the prompt to fit into a structured conversation template taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct on 15.05.2024. 
    Each message in the conversation is represented as a dictionary with 'role' and 'content'.
    """
    messages = []
    if sys_prompt:
        if "meta" in model_name:
            messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def prepare_interpretation(content: str) -> str:
    """
    Prepares a prompt for a Large Language Model to interpret an image and context jointly
    to extract the claim made by the post.
    """
    system_prompt = (
        "You are an AI assistant. Your task is to interpret the combination of an image "
        "and the accompanying text to extract the claim made by the post as accurately as possible. "
        "Analyze both the visual and textual information to understand the claim. "
    )

    prompt = f"USER: <image>\n{system_prompt} The textual part is: '{content}'\nASSISTANT:"
    return prompt
