import os.path
from typing import Collection, Any

from src.common.action import *
from src.common.claim import Claim
from src.common.content import Content
from src.common.document import FCDocument
from src.common.label import Label, DEFAULT_LABEL_DEFINITIONS
from src.common.results import Evidence, SearchResult
from src.utils.parsing import strip_string, remove_non_symbols

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


class JudgePrompt(Prompt):
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
        return read_md_file("src/prompts/judge.md")


class DecontextualizePrompt(Prompt):
    def __init__(self, claim: Claim):
        placeholder_targets = {
            "[ATOMIC_FACT]": claim.text,
            "[CONTEXT]": claim.original_context.text,  # TODO: improve this, add images etc.
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/decontextualize.md")


class FilterCheckWorthyPrompt(Prompt):
    def __init__(self, claim: Claim, filter: str = "default"):
        assert (filter in ["default", "custom"])
        placeholder_targets = {  # re-implement this
            "[SYMBOL]": SYMBOL,
            "[NOT_SYMBOL]": NOT_SYMBOL,
            "[ATOMIC_FACT]": claim,
            "[CONTEXT]": claim.original_context,
        }
        self.filter = filter
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        if self.filter == 'custom':
            return read_md_file("src/prompts/custom_checkworthy.md")
        else:
            return read_md_file("src/prompts/default_checkworthy.md")


class SummarizeResultPrompt(Prompt):
    def __init__(self, search_result: SearchResult, doc: FCDocument):
        placeholder_targets = {
            "[SEARCH_RESULT]": str(search_result),
            "[DOC]": str(doc),
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/summarize_result.md")


class SelectionPrompt(Prompt):
    def __init__(self, question: str, evidences: list[SearchResult]):
        placeholder_targets = {
            "[QUESTION]": question,
            "[EVIDENCES]": "\n\n".join(str(evidence) for evidence in evidences),
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/select_evidence.md")


class SummarizeDocPrompt(Prompt):
    def __init__(self, doc: FCDocument):
        super().__init__({"[DOC]": doc})

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/summarize_doc.md")


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
        # TODO: Check the Note files for the new actions
        if WikiDumpLookup in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/wiki_dump.md")
        elif WebSearch in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/web_search.md")
        elif DetectObjects in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/object_recognition.md")
        elif ReverseSearch in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/reverse_search.md")
        elif Geolocate in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/geo_location.md")
        elif FaceRecognition in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/face_recognition.md")
        elif CredibilityCheck in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/source_credibility_check.md")
        elif OCR in valid_actions:
            return read_md_file("src/prompts/plan_exemplars/ocr.md")
        else:
            return read_md_file("src/prompts/plan_exemplars/default.md")

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/plan.md")


class PoseQuestionsPrompt(Prompt):
    def __init__(self, doc: FCDocument, n_questions: int = 10):
        placeholder_targets = {
            "[DOC]": doc,
            "[N_QUESTIONS]": n_questions
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/pose_questions.md")


class ProposeQueries(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    def __init__(self, question: str, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/propose_queries.md")


class ProposeQuerySimple(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    def __init__(self, question: str):
        placeholder_targets = {
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/propose_query_simple.md")


class ProposeQueriesNoQuestions(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    def __init__(self, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/propose_queries_no_questions.md")


class AnswerCollectively(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    def __init__(self, question: str, results: list[SearchResult], doc: FCDocument):
        result_strings = [f"## Result `{i}`\n{str(result)}" for i, result in enumerate(results)]
        results_str = "\n\n".join(result_strings)

        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULTS]": results_str,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/answer_question_collectively.md")


class AnswerQuestion(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    def __init__(self, question: str, result: SearchResult):
        placeholder_targets = {
            "[QUESTION]": question,
            "[RESULT]": result,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/answer_question.md")


class AnswerQuestionNoEvidence(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    def __init__(self, question: str, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/answer_question_no_evidence.md")


class ReiteratePrompt(Prompt):  # TODO: Summarize each evidence instead of collection of all results
    def __init__(self, doc: FCDocument, evidences: Collection[Evidence]):
        results = []
        for evidence in evidences:
            results.extend(evidence.get_useful_results())
        results_str = "\n\n".join([str(r) for r in results])
        placeholder_targets = {
            "[DOC]": doc,
            "[RESULTS]": results_str,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/consolidate.md")


class InterpretPrompt(Prompt):
    def __init__(self, claim: Claim):
        placeholder_targets = {
            "[CLAIM]": claim,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/interpret.md")


class JudgeNaively(Prompt):
    def __init__(self, claim: Claim,
                 classes: list[Label],
                 class_definitions: dict[Label, str] = None):
        if class_definitions is None:
            class_definitions = DEFAULT_LABEL_DEFINITIONS
        class_str = '\n'.join([f"* `{cls.value}`: {remove_non_symbols(class_definitions[cls])}"
                               for cls in classes])
        placeholder_targets = {
            "[CLAIM]": claim,
            "[CLASSES]": class_str,
        }
        super().__init__(placeholder_targets)

    def assemble_prompt(self) -> str:
        return read_md_file("src/prompts/judge_naive.md")


def read_md_file(file_path: str) -> str:
    """Reads and returns the contents of the specified Markdown file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No Markdown file found at '{file_path}'.")
    with open(file_path, 'r') as f:
        return f.read()
