from abc import ABC

from common.label import Label
from common.utils import strip_string
from safe.prompts.common import STATEMENT_PLACEHOLDER, KNOWLEDGE_PLACEHOLDER


class Prompt(ABC):
    text: str
    placeholder_targets: dict[str, str] = {}

    def __init__(self):
        self.text = self.finalize_prompt()

    def finalize_prompt(self) -> str:
        """Turns a template prompt into a ready-to-send prompt string."""
        template = self.assemble_prompt()
        text = self.insert_variables(template)
        return strip_string(text)

    def assemble_prompt(self) -> str:
        """Collects and combines all pieces to form a template prompt, optionally
        containing placeholders to be replaced."""
        raise NotImplementedError()

    def insert_variables(self, text: str) -> str:
        for placeholder, keyword in self.placeholder_targets.items():
            text = text.replace(placeholder, keyword)
        return text

    def __str__(self):
        return self.text


class SearchPrompt(Prompt):
    def __init__(self, claim: str, knowledge: str, open_source: bool = False):
        self.placeholder_targets[STATEMENT_PLACEHOLDER] = claim
        self.placeholder_targets[KNOWLEDGE_PLACEHOLDER] = knowledge
        self.open_source = open_source
        super().__init__()

    def assemble_prompt(self) -> str:
        if self.open_source:
            return read_md_file("safe/prompts/search_google_open_source.md")
        else:
            return read_md_file("safe/prompts/search_google.md")


class ReasonPrompt(Prompt):
    # TODO: Add ICL
    # TODO: Add label choice
    placeholder_targets = {
        "[LABEL_SUPPORTED]": Label.SUPPORTED.value,
        "[LABEL_NEI]": Label.NEI.value,
        "[LABEL_REFUTED]": Label.REFUTED.value,
    }

    def __init__(self, claim: str, knowledge: str):
        self.placeholder_targets[STATEMENT_PLACEHOLDER] = claim
        self.placeholder_targets[KNOWLEDGE_PLACEHOLDER] = knowledge
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/reason.md")


def read_md_file(file_path: str) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    with open(file_path, 'r') as f:
        contents = f.read()
    return contents
