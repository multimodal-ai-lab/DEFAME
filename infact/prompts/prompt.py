import re
from typing import Collection, Any, Optional

from infact.common.medium import MultimediaSnippet
from infact.common.action import (OCR, ACTION_REGISTRY, FaceRecognition, WebSearch, WikiDumpLookup, DetectObjects,
                                  Geolocate, Action, CredibilityCheck, ReverseSearch)
from infact.common.claim import Claim
from infact.common.document import FCDocument
from infact.common.label import Label, DEFAULT_LABEL_DEFINITIONS
from infact.common.results import Evidence, SearchResult
from infact.utils.parsing import (strip_string, remove_non_symbols, extract_last_code_span, remove_code_blocks,
                                  extract_last_code_block, find_code_span, extract_last_paragraph, read_md_file,
                                  fill_placeholders)

SYMBOL = 'Check-worthy'
NOT_SYMBOL = 'Unimportant'


class Prompt(MultimediaSnippet):
    template_file_path: str

    def __init__(self,
                 placeholder_targets: dict[str, Any] = None,
                 text: str = None,
                 template_file_path: str = None):
        if text is None:
            text = self.compose_prompt(template_file_path, placeholder_targets)
        super().__init__(text)

    def compose_prompt(self, template_file_path: str = None,
                       placeholder_targets: dict[str, Any] = None) -> str:
        """Turns a template prompt into a ready-to-send prompt string."""
        template = self.get_template(template_file_path)
        if placeholder_targets is None:
            text = template
        else:
            text = fill_placeholders(template, placeholder_targets)
        return strip_string(text)

    def get_template(self, template_file_path: str = None) -> str:
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


class JudgePrompt(Prompt):
    template_file_path = "infact/prompts/judge.md"

    def __init__(self, doc: FCDocument,
                 classes: list[Label],
                 class_definitions: dict[Label, str] = None,
                 extra_rules: str = None):
        if class_definitions is None:
            class_definitions = DEFAULT_LABEL_DEFINITIONS
        self.classes = classes
        class_str = '\n'.join([f"* `{cls.value}`: {remove_non_symbols(class_definitions[cls])}"
                               for cls in classes])
        placeholder_targets = {
            "[DOC]": str(doc),
            "[CLASSES]": class_str,
            "[EXTRA_RULES]": "" if extra_rules is None else remove_non_symbols(extra_rules),
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        verdict = extract_verdict(response, classes=self.classes)
        if verdict is None:
            return None
        else:
            return dict(verdict=verdict, response=response)


def extract_verdict(response: str, classes: list[Label]) -> Optional[Label]:
    answer = extract_last_code_span(response)
    answer = re.sub(r'[^\w\-\s]', '', answer).strip().lower()

    if not answer:
        pattern = re.compile(r'\*\*(.*)\*\*', re.DOTALL)
        matches = pattern.findall(response) or ['']
        answer = matches[0]

    try:
        return Label(answer)

    except ValueError:
        # Maybe the label is a substring of the response
        for c in classes:
            if c.value in response:
                return c

    return None


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
        self.context = doc.claim.original_context
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

    def extract(self, response: str) -> dict | str | None:
        actions = self._extract_actions(response)
        reasoning = self._extract_reasoning(response)
        return dict(
            actions=actions,
            reasoning=reasoning,
            response=response,
        )

    def _extract_actions(self, answer: str) -> list[Action]:
        actions_str = extract_last_code_block(answer).replace("markdown", "")
        if not actions_str:
            candidates = []
            for action in ACTION_REGISTRY:
                pattern = re.compile(f'{action.name}("(.*?)")', re.DOTALL)
                candidates += pattern.findall(answer)
            actions_str = "\n".join(candidates)
        if not actions_str:
            # Potentially prompt LLM to correct format: Exprected format: action_name("query")
            return []
        raw_actions = actions_str.split('\n')
        actions = []
        for raw_action in raw_actions:
            action = self._parse_single_action(raw_action)
            if action:
                actions.append(action)
        return actions

    def _extract_reasoning(self, answer: str) -> str:
        return remove_code_blocks(answer).strip()

    def _parse_single_action(self, raw_action: str) -> Optional[Action]:
        if not raw_action:
            return None
        elif raw_action[0] == '"':
            raw_action = raw_action[1:]

        try:
            # Use regular expression to match action and argument in the form action(argument)
            match = re.match(r'(\w+)\((.*)\)', raw_action)

            # Extract action name and arguments
            if match:
                action_name, arguments = match.groups()
                arguments = arguments.strip()
            else:
                # self.logger.log(f"Invalid action format: {raw_action}")
                match = re.search(r'"(.*?)"', raw_action)
                arguments = f'"{match.group(1)}"' if match else f'"{raw_action}"'
                first_part = raw_action.split(' ')[0]
                action_name = re.sub(r'[^a-zA-Z0-9_]', '', first_part)

            if "image" in arguments:
                images = self.context.images
                # TODO: implement multi image argument
                arguments = images[0]

            for action in ACTION_REGISTRY:
                if action_name == action.name:
                    return action(arguments)

            raise ValueError(f'Invalid action format: {raw_action} . Expected format: action_name("query")')

        except Exception as e:
            # self.logger.log(f"WARNING: Failed to parse '{raw_action}':\n{e}")
            pass

        return None


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

    def extract(self, response: str) -> dict | str | None:
        questions = find_code_span(response)
        return dict(
            questions=questions,
            response=response,
        )


class ProposeQueries(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_queries.md"

    def __init__(self, question: str, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


def extract_queries(response: str) -> list[WebSearch]:
    matches = find_code_span(response)
    queries = []
    for match in matches:
        query = strip_string(match)
        action = WebSearch(f'"{query}"')
        queries.append(action)
    return queries


class ProposeQuerySimple(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_query_simple.md"

    def __init__(self, question: str):
        placeholder_targets = {
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


class ProposeQueriesNoQuestions(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_queries_no_questions.md"

    def __init__(self, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


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

    def extract(self, response: str) -> dict | str | None:
        """Extract result ID and answer to the question from response"""
        answered = "NONE" not in response and "None" not in response

        out = dict(
            answered=answered,
            response=response,
        )

        if answered:
            result_id = extract_last_code_span(response)
            if result_id != "":
                result_id = int(result_id)
                answer = extract_last_paragraph(response)
                out.update(dict(
                    answer=answer,
                    result_id=result_id,
                ))

        return out


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

    def extract(self, response: str) -> dict | str | None:
        """Extract result ID and answer to the question from response"""
        answered = "NONE" not in response and "None" not in response

        out = dict(
            answered=answered,
            response=response,
        )

        if answered:
            answer = extract_last_paragraph(response)
            out.update(dict(answer=answer))

        return out


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
        self.classes = classes
        if class_definitions is None:
            class_definitions = DEFAULT_LABEL_DEFINITIONS
        class_str = '\n'.join([f"* `{cls.value}`: {remove_non_symbols(class_definitions[cls])}"
                               for cls in classes])
        placeholder_targets = {
            "[CLAIM]": claim,
            "[CLASSES]": class_str,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        verdict = extract_verdict(response, classes=self.classes)
        if verdict is None:
            return None
        else:
            return dict(verdict=verdict, response=response)
