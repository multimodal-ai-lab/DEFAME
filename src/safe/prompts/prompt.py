from abc import ABC

from common.label import Label
from common.utils import strip_string


SYMBOL = 'Check-worthy'
NOT_SYMBOL = 'Unimportant'


class Prompt(ABC):
    text: str
    placeholder_targets: dict[str, str] = {}

    def __init__(self):
        self.text = self.finalize_prompt()

    def finalize_prompt(self) -> str:
        """Turns a template prompt into a ready-to-send prompt string."""
        template = self.assemble_prompt()
        text = self.insert_into_placeholders(template)
        return strip_string(text)

    def assemble_prompt(self) -> str:
        """Collects and combines all pieces to form a template prompt, optionally
        containing placeholders to be replaced."""
        raise NotImplementedError()

    def insert_into_placeholders(self, text: str) -> str:
        """Replaces all specified placeholders in placeholder_targets with the
        respective target content."""
        for placeholder, target in self.placeholder_targets.items():
            text = text.replace(placeholder, target)
        return text

    def __str__(self):
        return self.text


class SearchPrompt(Prompt):
    def __init__(self, claim: str, knowledge: str, past_queries: str,
                 search_engine: str = "google", open_source: bool = False):
        self.placeholder_targets["[STATEMENT]"] = claim
        self.placeholder_targets["[KNOWLEDGE]"] = knowledge
        self.placeholder_targets["[PAST_QUERIES]"] = past_queries
        self.open_source = open_source
        assert search_engine in ["google", "wiki"]
        self.search_engine = search_engine
        super().__init__()

    def assemble_prompt(self) -> str:
        match self.search_engine:
            case "google":
                if self.open_source:
                    return read_md_file("safe/prompts/search_google_open_source.md")
                else:
                    return read_md_file("safe/prompts/search_google.md")
            case "wiki":
                return read_md_file("safe/prompts/search_wiki_dump.md")


class ReasonPrompt(Prompt):
    # TODO: Add ICL
    # TODO: Add label choice
    # TODO: Add 'contradicting' label
    placeholder_targets = {
        "[LABEL_SUPPORTED]": Label.SUPPORTED.value,
        "[LABEL_NEI]": Label.NEI.value,
        "[LABEL_REFUTED]": Label.REFUTED.value,
    }

    def __init__(self, claim: str, knowledge: str):
        self.placeholder_targets["[STATEMENT]"] = claim
        self.placeholder_targets["[KNOWLEDGE]"] = knowledge
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/reason.md")


class DecontextualizePrompt(Prompt):
    def __init__(self, atomic_fact: str, context: str):
        self.placeholder_targets["[ATOMIC_FACT]"] = atomic_fact
        self.placeholder_targets["[CONTEXT]"] = context
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/decontextualize.md")


class FilterCheckWorthyPrompt(Prompt):
    placeholder_targets = {
        "[SYMBOL]": SYMBOL,
        "[NOT_SYMBOL]": NOT_SYMBOL,
    }

    def __init__(self, atomic_fact: str, context: str):
        self.placeholder_targets["[ATOMIC_FACT]"] = atomic_fact
        self.placeholder_targets["[CONTEXT]"] = context
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/filter_check_worthy.md")


class SummarizePrompt(Prompt):
    def __init__(self, query: str, search_result: str):
        self.placeholder_targets["[QUERY]"] = query
        self.placeholder_targets["[SEARCH_RESULT]"] = search_result[:50000]  # Cut to avoid hitting the context window limit
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/summarize.md")


def read_md_file(file_path: str) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    with open(file_path, 'r') as f:
        contents = f.read()
    return contents
