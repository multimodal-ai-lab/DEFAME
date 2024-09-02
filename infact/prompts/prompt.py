from typing import Collection, Any, Sequence

from infact.common.action import *
from infact.common.claim import Claim
from infact.common.document import FCDocument
from infact.common.label import Label, DEFAULT_LABEL_DEFINITIONS
from infact.common.results import Evidence, SearchResult
from infact.utils.parsing import strip_string, remove_non_symbols
from infact.utils.parsing import read_md_file
from PIL.Image import Image

SYMBOL = 'Check-worthy'
NOT_SYMBOL = 'Unimportant'


class Prompt:
    template_file_path: str

    def __init__(self,
                 placeholder_targets: dict[str, Any] = None,
                 text: str = None,
                 images: list[Image] = None,
                 template_file_path: str = None):
        if template_file_path is not None:
            self.template_file_path = template_file_path

        if placeholder_targets is not None:
            self.text: str = self.compose_prompt(placeholder_targets)
        else:
            assert text is not None
            self.text = text

        self.images = images

    def compose_prompt(self, placeholder_targets: dict[str, Any]) -> str:
        """Turns a template prompt into a ready-to-send prompt string."""
        template = self.get_template()
        text = self.insert_into_placeholders(template, placeholder_targets)
        return strip_string(text)

    def get_template(self) -> str:
        """Collects and combines all pieces to form a template prompt, optionally
        containing placeholders to be replaced."""
        assert self.template_file_path is not None
        return read_md_file(self.template_file_path)

    def insert_into_placeholders(self, text: str, placeholder_targets: dict[str, Any]) -> str:
        """Replaces all specified placeholders in placeholder_targets with the
        respective target content."""
        for placeholder, target in placeholder_targets.items():
            if placeholder not in text:
                raise ValueError(f"Placeholder '{placeholder}' not found in prompt template:\n{text}")
            text = text.replace(placeholder, str(target))
        return text

    def is_multimodal(self):
        return isinstance(self.images, Sequence) and len(self.images) > 0

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.__str__())


class JudgePrompt(Prompt):
    template_file_path = "infact/prompts/judge.md"

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


class DecontextualizePrompt(Prompt):
    template_file_path = "infact/prompts/decontextualize.md"

    def __init__(self, claim: Claim):
        placeholder_targets = {
            "[ATOMIC_FACT]": claim.text,
            "[CONTEXT]": claim.original_context.text,  # TODO: improve this, add images etc.
        }
        super().__init__(placeholder_targets)


class FilterCheckWorthyPrompt(Prompt):
    def __init__(self, claim: Claim, filter_method: str = "default"):
        assert (filter_method in ["default", "custom"])
        placeholder_targets = {  # re-implement this
            "[SYMBOL]": SYMBOL,
            "[NOT_SYMBOL]": NOT_SYMBOL,
            "[ATOMIC_FACT]": claim,
            "[CONTEXT]": claim.original_context,
        }
        if filter_method == "custom":
            self.template_file_path = "infact/prompts/custom_checkworthy.md"
        else:
            self.template_file_path = "infact/prompts/default_checkworthy.md"
        super().__init__(placeholder_targets)


class SummarizeResultPrompt(Prompt):
    template_file_path = "infact/prompts/summarize_result.md"

    def __init__(self, search_result: SearchResult, doc: FCDocument):
        placeholder_targets = {
            "[SEARCH_RESULT]": str(search_result),
            "[DOC]": str(doc),
        }
        super().__init__(placeholder_targets)


class SelectionPrompt(Prompt):
    template_file_path = "infact/prompts/select_evidence.md"

    def __init__(self, question: str, evidences: list[SearchResult]):
        placeholder_targets = {
            "[QUESTION]": question,
            "[EVIDENCES]": "\n\n".join(str(evidence) for evidence in evidences),
        }
        super().__init__(placeholder_targets)


class SummarizeDocPrompt(Prompt):
    template_file_path = "infact/prompts/summarize_doc.md"

    def __init__(self, doc: FCDocument):
        super().__init__({"[DOC]": doc})


class PlanPrompt(Prompt):
    template_file_path = "infact/prompts/plan.md"

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
            return read_md_file("infact/prompts/plan_exemplars/wiki_dump.md")
        elif WebSearch in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/web_search.md")
        elif DetectObjects in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/object_recognition.md")
        elif ReverseSearch in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/reverse_search.md")
        elif Geolocate in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/geo_location.md")
        elif FaceRecognition in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/face_recognition.md")
        elif CredibilityCheck in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/source_credibility_check.md")
        elif OCR in valid_actions:
            return read_md_file("infact/prompts/plan_exemplars/ocr.md")
        else:
            return read_md_file("infact/prompts/plan_exemplars/default.md")


class PoseQuestionsPrompt(Prompt):
    def __init__(self, doc: FCDocument, n_questions: int = 10, interpret: bool = True):
        placeholder_targets = {
            "[CLAIM]": doc.claim,
            "[N_QUESTIONS]": n_questions
        }
        if interpret:
            self.template_file_path = "infact/prompts/pose_questions.md"
        else:
            self.template_file_path = "infact/prompts/pose_questions_no_interpretation.md"
        super().__init__(placeholder_targets)


class ProposeQueries(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_queries.md"

    def __init__(self, question: str, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)


class ProposeQuerySimple(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_query_simple.md"

    def __init__(self, question: str):
        placeholder_targets = {
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)


class ProposeQueriesNoQuestions(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_queries_no_questions.md"

    def __init__(self, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
        }
        super().__init__(placeholder_targets)


class AnswerCollectively(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    template_file_path = "infact/prompts/answer_question_collectively.md"

    def __init__(self, question: str, results: list[SearchResult], doc: FCDocument):
        result_strings = [f"## Result `{i}`\n{str(result)}" for i, result in enumerate(results)]
        results_str = "\n\n".join(result_strings)

        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULTS]": results_str,
        }
        super().__init__(placeholder_targets)


class AnswerQuestion(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    template_file_path = "infact/prompts/answer_question.md"

    def __init__(self, question: str, result: SearchResult, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULT]": result,
        }
        super().__init__(placeholder_targets)


class AnswerQuestionNoEvidence(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    template_file_path = "infact/prompts/answer_question_no_evidence.md"

    def __init__(self, question: str, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)


class ReiteratePrompt(Prompt):  # TODO: Summarize each evidence instead of collection of all results
    template_file_path = "infact/prompts/consolidate.md"

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


class InterpretPrompt(Prompt):
    template_file_path = "infact/prompts/interpret.md"

    def __init__(self, claim: Claim):
        placeholder_targets = {
            "[CLAIM]": claim,
        }
        super().__init__(placeholder_targets)


class JudgeNaively(Prompt):
    template_file_path = "infact/prompts/judge_naive.md"

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
