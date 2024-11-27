from typing import Any
from pathlib import Path

from defame.common.medium import MultimediaSnippet
from defame.utils.parsing import (strip_string, read_md_file,
                                  fill_placeholders)


class Prompt(MultimediaSnippet):
    template_file_path: str
    name: str = "DefaultPrompt"
    retry_instruction: str

    def __init__(self,
                 placeholder_targets: dict[str, Any] = None,
                 name: str = None,
                 text: str = None,
                 template_file_path: str | Path = None):
        if name:
            self.name = name
        if text is None:
            text = self.compose_prompt(template_file_path, placeholder_targets)
        super().__init__(text)

    def compose_prompt(self, template_file_path: str | Path = None,
                       placeholder_targets: dict[str, Any] = None) -> str:
        """Turns a template prompt into a ready-to-send prompt string."""
        template = self.get_template(template_file_path)
        if placeholder_targets is None:
            text = template
        else:
            text = fill_placeholders(template, placeholder_targets)
        return strip_string(text)

    def get_template(self, template_file_path: str | Path = None) -> str:
        """Collects and combines all pieces to form a template prompt, optionally
        containing placeholders to be replaced."""
        if template_file_path is None:
            assert self.template_file_path is not None
            template_file_path = self.template_file_path
        return read_md_file(template_file_path)

    def extract(self, response: str) -> dict | str | None:
        """Takes the model's output string and extracts the expected data.
        Returns the data as a dictionary."""
        return response  # default implementation

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.__str__())
