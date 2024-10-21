import re
from pathlib import Path
from typing import Collection, Optional

from infact.common import FCDocument, Label, Claim, Action, Evidence, Prompt, Content
from infact.common.label import DEFAULT_LABEL_DEFINITIONS
from infact.common.misc import WebSource
from infact.common.results import Result
from infact.utils.parsing import (remove_non_symbols, extract_last_code_span, read_md_file,
                                  find_code_span, extract_last_paragraph, extract_last_code_block,
                                  strip_string, remove_code_blocks)

SYMBOL = 'Check-worthy'
NOT_SYMBOL = 'Unimportant'


class JudgePrompt(Prompt):
    template_file_path = "infact/prompts/judge.md"

    def __init__(self, doc: FCDocument,
                 classes: Collection[Label],
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

    def extract(self, response: str) -> dict:
        verdict = extract_verdict(response, classes=self.classes)
        return dict(verdict=verdict, response=response)


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

    def __init__(self, search_result: WebSource, doc: FCDocument):
        placeholder_targets = {
            "[SEARCH_RESULT]": str(search_result),
            "[DOC]": str(doc),
        }
        super().__init__(placeholder_targets)

class SummarizeManipulationResultPrompt(Prompt):
    template_file_path = "infact/prompts/summarize_manipulation_result.md"

    def __init__(self, manipulation_result: Result):
        placeholder_targets = {
            "[MANIPULATION_RESULT]": str(manipulation_result),
        }
        super().__init__(placeholder_targets)


class SelectionPrompt(Prompt):
    template_file_path = "infact/prompts/select_evidence.md"

    def __init__(self, question: str, evidences: list[WebSource]):
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
            "[EXEMPLARS]": load_exemplars(valid_actions),
            "[EXTRA_RULES]": extra_rules,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict:
        # In case "image:k is referenced by the LLM by mistake"
        original_response = response
        claim_image = self.images[0].reference
        pattern = re.compile(r'<image:[a-zA-Z0-9_]+>')
        multimodal_actions = pattern.findall(response)

        if multimodal_actions:
            response = pattern.sub(claim_image, response)
            print(f"WARNING: <image:k> was replaced by {claim_image} to generate response: {response}")

        actions = extract_actions(response)
        reasoning = extract_reasoning(response)
        return dict(
            actions=actions,
            reasoning=reasoning,
            response=response,
        )


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

    def extract(self, response: str) -> dict:
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

    def extract(self, response: str) -> dict:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


class ProposeQuerySimple(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "infact/prompts/propose_query_simple.md"

    def __init__(self, question: str):
        placeholder_targets = {
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict:
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

    def extract(self, response: str) -> dict:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


class AnswerCollectively(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    template_file_path = "infact/prompts/answer_question_collectively.md"

    def __init__(self, question: str, results: list[WebSource], doc: FCDocument):
        result_strings = [f"## Result `{i}`\n{str(result)}" for i, result in enumerate(results)]
        results_str = "\n\n".join(result_strings)

        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULTS]": results_str,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict:
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

    def __init__(self, question: str, result: WebSource, doc: FCDocument):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULT]": result,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict:
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


class ReiteratePrompt(Prompt):
    template_file_path = "infact/prompts/consolidate.md"

    def __init__(self, doc: FCDocument, evidences: Collection[Evidence]):
        evidence_str = "\n\n".join([str(e) for e in evidences])
        placeholder_targets = {
            "[DOC]": doc,
            "[RESULTS]": evidence_str,
        }
        super().__init__(placeholder_targets)


class InterpretPrompt(Prompt):
    template_file_path = "infact/prompts/interpret.md"

    def __init__(self, content: Content, guidelines: str = ''):
        if guidelines:
            guidelines = "# Guidelines\n" + guidelines
        placeholder_targets = {
            "[CONTENT]": content,
            "[GUIDELINES]": guidelines,
        }
        super().__init__(placeholder_targets)

    def extract(self, response: str) -> dict:
        answer = extract_last_code_span(response)
        answer = re.sub(r'[^\w\-\s]', '', answer).strip().lower()
        out = dict(
            answer=answer,
            response=response,
        )
        if not answer:
            pattern = re.compile(r'\*\*(.*)\*\*', re.DOTALL)
            matches = pattern.findall(response) or ['']
            answer = matches[0]
            out.update(dict(answer=answer))
        return out


class JudgeNaively(Prompt):
    template_file_path = "infact/prompts/judge_naive.md"

    def __init__(self, claim: Claim,
                 classes: Collection[Label],
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

    def extract(self, response: str) -> dict:
        verdict = extract_verdict(response, classes=self.classes)
        return dict(verdict=verdict, response=response)


def load_exemplars(valid_actions: list[type[Action]]) -> str:
    exemplars_dir = Path("infact/prompts/plan_exemplars")
    exemplar_paths = []
    for a in valid_actions:
        exemplar_path = exemplars_dir / f"{a.name}.md"
        if exemplar_path.exists():
            exemplar_paths.append(exemplar_path)

    if len(exemplar_paths) == 0:
        return read_md_file(exemplars_dir / "default.md")
    else:
        return "\n\n".join([read_md_file(path) for path in exemplar_paths])


def parse_single_action(raw_action: str) -> Optional[Action]:
    from infact.tools import ACTION_REGISTRY

    if not raw_action:
        return None
    elif raw_action[0] == '"':
        raw_action = raw_action[1:]

    try:
        match = re.match(r'(\w+)\((.*)\)', raw_action)
        if match:
            action_name, arguments = match.groups()
            arguments = arguments.strip()
        else:
            match = re.search(r'"(.*?)"', raw_action)
            arguments = f'"{match.group(1)}"' if match else f'"{raw_action}"'
            first_part = raw_action.split(' ')[0]
            action_name = re.sub(r'[^a-zA-Z0-9_]', '', first_part)

        for action in ACTION_REGISTRY:
            if action_name == action.name:
                return action(arguments)

        raise ValueError(f'Invalid action: {raw_action}\nExpected format: action_name(<arg1>, <arg2>, ...)')

    except Exception as e:
        # TODO: Make logger global and enable the following log:
        # self.logger.log(f"WARNING: Failed to parse '{raw_action}':\n{e}")
        print(f"WARNING: Failed to parse '{raw_action}':\n{e}")
        pass

    return None


def extract_actions(answer: str, limit=5) -> list[Action]:
    from infact.tools import ACTION_REGISTRY

    actions_str = extract_last_code_block(answer).replace("markdown", "")
    if not actions_str:
        candidates = []
        for action in ACTION_REGISTRY:
            pattern = re.compile(rf'({re.escape(action.name)}\(.+?\))', re.DOTALL)
            candidates += pattern.findall(answer)
        actions_str = "\n".join(candidates)
    if not actions_str:
        # Potentially prompt LLM to correct format: Expected format: action_name("arguments")
        return []
    raw_actions = actions_str.split('\n')
    actions = []
    for raw_action in raw_actions:
        action = parse_single_action(raw_action)
        if action:
            actions.append(action)
        if len(actions) == limit:
            break
    return actions


def extract_verdict(response: str, classes: Collection[Label]) -> Optional[Label]:
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


def extract_queries(response: str) -> list:
    from infact.tools import WebSearch
    matches = find_code_span(response)
    queries = []
    for match in matches:
        query = strip_string(match)
        action = WebSearch(f'"{query}"')
        queries.append(action)
    return queries


def extract_reasoning(answer: str) -> str:
    return remove_code_blocks(answer).strip()
