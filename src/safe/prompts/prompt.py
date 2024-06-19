import os.path
from abc import ABC
from typing import Sequence, Collection, Any

from common.label import Label, DEFAULT_LABEL_DEFINITIONS
from common.utils import strip_string, remove_non_symbols
from common.shared_config import search_engine_options
from common.document import FCDocument
from common.action import Action, WikiDumpLookup, WebSearch
from common.results import Result, SearchResult
from common.claim import Claim

SYMBOL = 'Check-worthy'
NOT_SYMBOL = 'Unimportant'


class Prompt(ABC):
    def __init__(self, placeholder_targets: dict[str, Any]):
        self.placeholder_targets = placeholder_targets
        self.text: str = self.finalize_prompt()

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
            if placeholder not in text:
                raise ValueError(f"Placeholder '{placeholder}' not found in prompt template:\n{text}")
            text = text.replace(placeholder, str(target))
        return text

    def __str__(self):
        return self.text


class SearchPrompt(Prompt):
    # TODO Keep in mind that the current Search Prompt does not use the [PAST_QUERIES] placeholder.
    # TODO: Rework search prompt to also consider previous searching reasoning

    def __init__(self, claim: str, knowledge: str, past_queries: str,
                 search_engine: str = "google", open_source: bool = False):
        placeholder_targets = {
            "[STATEMENT]": claim,
            "[KNOWLEDGE]": knowledge,
            "[PAST_QUERIES]": past_queries,
        }
        self.open_source = open_source
        assert search_engine in search_engine_options
        self.search_engine = search_engine
        super().__init__(placeholder_targets)

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
    def __init__(self, doc: FCDocument,
                 classes: list[Label],
                 class_definitions: dict[Label, str] = None,
                 extra_rules: str = None):
        if class_definitions is None:
            class_definitions = DEFAULT_LABEL_DEFINITIONS
        class_str = '\n'.join([f"* `{cls.value}`: {remove_non_symbols(class_definitions[cls])}"
                               for cls in classes])
        placeholder_targets = {
            "[DOC]": str(doc),
            "[CLASSES]": class_str,
            "[EXTRA_RULES]": "" if extra_rules is None else remove_non_symbols(extra_rules),
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/judge.md")


class DecontextualizePrompt(Prompt):
    def __init__(self, atomic_fact: str, context: str):
        placeholder_targets = {
            "[ATOMIC_FACT]": atomic_fact,
            "[CONTEXT]": context,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/decontextualize.md")


class FilterCheckWorthyPrompt(Prompt):

    def __init__(self, atomic_fact: str, context: str, filter: str = "default"):
        assert (filter in ["default", "custom"])
        placeholder_targets = {
            "[SYMBOL]": SYMBOL,
            "[NOT_SYMBOL]": NOT_SYMBOL,
            "[ATOMIC_FACT]": atomic_fact,
            "[CONTEXT]": context,
        }
        self.filter = filter
        super().__init__(placeholder_targets)

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
        placeholder_targets = {
            "[SEARCH_RESULT]": str(search_result),
            "[DOC]": str(doc),
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/summarize_result.md")


class SummarizeDocPrompt(Prompt):
    def __init__(self, doc: FCDocument):
        super().__init__({"[DOC]": doc})

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/summarize_doc.md")


class PlanPrompt(Prompt):
    def __init__(self, doc: FCDocument,
                 valid_actions: list[type[Action]],
                 extra_rules: str = None):
        valid_action_str = "\n\n".join([f"* `{a.name}`\n"
                                        f"   * Description: {remove_non_symbols(a.description)}\n"
                                        f"   * How to use: {remove_non_symbols(a.how_to)}\n"
                                        f"   * Format: {a.format}" for a in valid_actions])
        extra_rules = "" if extra_rules is None else remove_non_symbols(extra_rules)
        placeholder_targets = {
            "[DOC]": doc,
            "[VALID_ACTIONS]": valid_action_str,
            "[EXEMPLARS]": self.load_exemplars(valid_actions),
            "[EXTRA_RULES]": extra_rules,
        }
        super().__init__(placeholder_targets)

    def load_exemplars(self, valid_actions) -> str:
        if WikiDumpLookup in valid_actions:
            return read_md_file("safe/prompts/plan_exemplars/wiki_dump.md")
        elif WebSearch in valid_actions:
            return read_md_file("safe/prompts/plan_exemplars/web_search.md")
        else:
            return read_md_file("safe/prompts/plan_exemplars/default.md")

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/plan.md")


class PreparePrompt(Prompt):
    def __init__(self, claim: Claim, extra_rules: str = None):
        placeholder_targets = {
            "[CLAIM]": claim,
            "[EXTRA_RULES]": "" if extra_rules is None else remove_non_symbols(extra_rules),
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/initial_reason.md")


class ReiteratePrompt(Prompt):
    def __init__(self, doc: FCDocument, results: Collection[Result]):
        results_str = "\n\n".join([str(r) for r in results if r.is_useful()])
        placeholder_targets = {
            "[DOC]": doc,
            "[RESULTS]": results_str,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("safe/prompts/consolidate.md")


def read_md_file(file_path: str) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No Markdown file found at '{file_path}'.")
    with open(file_path, 'r') as f:
        return f.read()
