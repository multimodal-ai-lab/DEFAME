from abc import ABC
from typing import Sequence

from common.label import Label, DEFAULT_LABEL_DEFINITIONS
from common.utils import strip_string, remove_non_letters
from common.shared_config import search_engine_options
from common.document import FCDocument
from common.action import Action, WebSearch, WikiDumpLookup
from common.results import SearchResult

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
    # TODO Keep in mind that the current Search Prompt does not use the [PAST_QUERIES] placeholder.
    # TODO: Rework search prompt to also consider previous searching reasoning

    def __init__(self, claim: str, knowledge: str, past_queries: str,
                 search_engine: str = "google", open_source: bool = False):
        self.placeholder_targets["[STATEMENT]"] = claim
        self.placeholder_targets["[KNOWLEDGE]"] = knowledge
        self.placeholder_targets["[PAST_QUERIES]"] = past_queries
        self.open_source = open_source
        assert search_engine in search_engine_options
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
            case "duckduck":
                return read_md_file("safe/prompts/search_google_open_source.md")
            case _:
                return read_md_file("safe/prompts/search_default.md")


class ReasonPrompt(Prompt):
    # TODO: Add ICL
    def __init__(self, claim: str, knowledge: str, classes: Sequence[Label]):
        label_options_string = (
                'According to your reasoning, your final answer should be either '
                + ', '.join(f'"{cls.value}"' for cls in classes[:-1])
                + f', or "{classes[-1].value}". Wrap your final answer in square brackets.'
        )

        # Add the LABEL_OPTIONS entry to the placeholder_targets dictionary
        self.placeholder_targets["[LABEL_OPTIONS]"] = label_options_string
        self.placeholder_targets["[STATEMENT]"] = claim
        self.placeholder_targets["[KNOWLEDGE]"] = knowledge
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/reason.md")


class JudgePrompt(Prompt):
    # TODO: Add ICL
    def __init__(self, doc: FCDocument, classes: list[Label], class_definitions: dict[Label, str] = None):
        if class_definitions is None:
            class_definitions = DEFAULT_LABEL_DEFINITIONS
        class_str = '\n'.join([f"* `{cls.value}`: {remove_non_letters(class_definitions[cls])}"
                               for cls in classes])
        self.placeholder_targets["[DOC]"] = str(doc)
        self.placeholder_targets["[CLASSES]"] = class_str
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/judge.md")


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

    def __init__(self, atomic_fact: str, context: str, filter: str = "default"):
        assert (filter in ["default", "custom"])
        self.filter = filter
        self.placeholder_targets["[ATOMIC_FACT]"] = atomic_fact
        self.placeholder_targets["[CONTEXT]"] = context
        super().__init__()

    def assemble_prompt(self) -> str:
        if self.filter == 'custom':
            return read_md_file("safe/prompts/custom_checkworthy.md")
        else:
            return read_md_file("safe/prompts/default_checkworthy.md")


class SummarizePrompt(Prompt):
    def __init__(self, claim: str, query: str, search_result: str):
        self.placeholder_targets["[STATEMENT]"] = claim
        self.placeholder_targets["[QUERY]"] = query
        self.placeholder_targets["[SEARCH_RESULT]"] = search_result
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/summarize.md")


class SummarizeResultPrompt(Prompt):
    def __init__(self, search_result: SearchResult, doc: FCDocument):
        search_result_str = f"From {search_result.source}:\n{search_result.text}"
        self.placeholder_targets["[SEARCH_RESULT]"] = search_result_str
        context_str = f'CLAIM: "{doc.claim}"\n\nREASONING:\n' + "\n".join(doc.get_all_reasoning())
        self.placeholder_targets["[DOC]"] = context_str
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/summarize_result.md")


class SummarizeDocPrompt(Prompt):
    def __init__(self, doc: FCDocument):
        self.placeholder_targets["[DOC]"] = str(doc)
        super().__init__()

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/summarize_doc.md")


class PlanPrompt(Prompt):
    def __init__(self, doc: FCDocument, valid_actions: list[type[Action]]):
        valid_action_str = "\n\n".join([f"* `{a.name}`\n"
                                        f"   * Description: {remove_non_letters(a.description)}\n"
                                        f"   * How to use: {remove_non_letters(a.how_to)}\n"
                                        f"   * Format: {a.format}" for a in valid_actions])
        self.placeholder_targets["[DOC]"] = str(doc)
        self.placeholder_targets["[VALID_ACTIONS]"] = valid_action_str
        self.placeholder_targets["[EXEMPLARS]"] = self.load_exemplars(valid_actions)
        super().__init__()

    def load_exemplars(self, valid_actions) -> str:
        if WikiDumpLookup in valid_actions:
            return read_md_file("safe/prompts/plan_exemplars/wiki_dump.md")
        else:
            return read_md_file("safe/prompts/plan_exemplars/default.md")

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/plan.md")


def read_md_file(file_path: str) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    with open(file_path, 'r') as f:
        contents = f.read()
    return contents
