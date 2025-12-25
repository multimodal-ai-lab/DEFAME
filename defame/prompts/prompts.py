import re
import traceback
from pathlib import Path
from typing import Collection, Optional
from ezmm import MultimodalSequence

from defame.common import Report, Label, Claim, Action, Prompt, Content, logger
from defame.common.action import get_action_documentation
from defame.common.label import DEFAULT_LABEL_DEFINITIONS
from defame.evidence_retrieval.integrations import SocialMediaPost
from defame.evidence_retrieval.integrations.search.common import Source
from defame.common.results import Results
from defame.evidence_retrieval.integrations.social_media.common import SocialMediaClaim
from defame.utils.parsing import (remove_non_symbols, extract_last_code_span, read_md_file,
                                  find_code_span, extract_last_paragraph, extract_last_python_code_block,
                                  strip_string, remove_code_blocks, parse_function_call)

SYMBOL = 'Check-worthy'
NOT_SYMBOL = 'Unimportant'


class JudgePrompt(Prompt):
    template_file_path = "defame/prompts/judge.md"
    retry_instruction = ("(Do not forget to choose one option from Decision Options "
                         "and enclose it in backticks like `this`)")

    def __init__(self, doc: Report,
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
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        verdict = extract_verdict(response, classes=self.classes)
        if verdict is None:
            return None
        else:
            return dict(verdict=verdict, response=response)


class DecontextualizePrompt(Prompt):
    template_file_path = "defame/prompts/decontextualize.md"

    def __init__(self, claim: Claim):
        placeholder_targets = {
            "[ATOMIC_FACT]": str(claim),
            "[CONTEXT]": str(claim.context),
        }
        super().__init__(placeholder_targets=placeholder_targets)


class FilterCheckWorthyPrompt(Prompt):
    def __init__(self, claim: Claim, filter_method: str = "default"):
        assert (filter_method in ["default", "custom"])
        placeholder_targets = {  # re-implement this
            "[SYMBOL]": SYMBOL,
            "[NOT_SYMBOL]": NOT_SYMBOL,
            "[ATOMIC_FACT]": claim,
            "[CONTEXT]": claim.context,
        }
        if filter_method == "custom":
            self.template_file_path = "defame/prompts/custom_checkworthy.md"
        else:
            self.template_file_path = "defame/prompts/default_checkworthy.md"
        super().__init__(placeholder_targets=placeholder_targets)


class SummarizeSourcePrompt(Prompt):
    template_file_path = "defame/prompts/summarize_source.md"

    def __init__(self, source: Source, doc: Report):
        placeholder_targets = {
            "[SOURCE]": str(source),
            "[DOC]": str(doc),
        }
        super().__init__(placeholder_targets=placeholder_targets)


class SummarizeManipulationResultPrompt(Prompt):
    template_file_path = "defame/prompts/summarize_manipulation_result.md"

    def __init__(self, manipulation_result: Results):
        placeholder_targets = {
            "[MANIPULATION_RESULT]": str(manipulation_result),
        }
        super().__init__(placeholder_targets=placeholder_targets)


class SummarizeDocPrompt(Prompt):
    template_file_path = "defame/prompts/summarize_doc.md"

    def __init__(self, doc: Report):
        super().__init__(placeholder_targets={"[DOC]": doc})


class PlanPrompt(Prompt):
    template_file_path = "defame/prompts/plan.md"

    def __init__(self, doc: Report,
                 valid_actions: Collection[type[Action]],
                 extra_rules: str = None,
                 all_actions: bool = False):
        action_docs = [get_action_documentation(a) for a in valid_actions]
        valid_action_str = "\n\n".join(action_docs)
        extra_rules = "" if extra_rules is None else remove_non_symbols(extra_rules)
        if all_actions:
            extra_rules = "Very Important: No need to be frugal. Choose all available actions at least once."

        placeholder_targets = {
            "[DOC]": doc,
            "[VALID_ACTIONS]": valid_action_str,
            # "[EXEMPLARS]": load_exemplars(valid_actions),
            "[EXTRA_RULES]": extra_rules,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        # TODO: Prevent the following from happening at all.
        # It may accidentally happen that the LLM generated "<image:k>" in its response (because it was
        # included as an example in the prompt).
        pattern = re.compile(r'<image:[a-z]>')
        matches = pattern.findall(response)

        if matches:
            # Replace "<image:k>" with the reference to the claim's image by assuming that the first image
            # is tha claim image.
            if self.images:
                claim_image_ref = self.images[
                    0].reference  # Be careful that the Plan Prompt always has the Claim image first before any other image!
                response = pattern.sub(claim_image_ref, response)
                logger.warning(f"LLM generated reference '<image:k>'. Replacing it by {claim_image_ref}.")

        actions = extract_actions(response)
        reasoning = extract_reasoning(response)
        return dict(
            actions=actions,
            reasoning=reasoning,
            response=response,
        )


class PoseQuestionsPrompt(Prompt):
    def __init__(self, doc: Report, n_questions: int = 10, interpret: bool = True):
        placeholder_targets = {
            "[CLAIM]": doc.claim,
            "[N_QUESTIONS]": n_questions
        }
        if interpret:
            self.template_file_path = "defame/prompts/pose_questions.md"
        else:
            self.template_file_path = "defame/prompts/pose_questions_no_interpretation.md"
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        questions = find_code_span(response)
        return dict(
            questions=questions,
            response=response,
        )


class ProposeQueries(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "defame/prompts/propose_queries.md"

    def __init__(self, question: str, doc: Report):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


class ProposeQuerySimple(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "defame/prompts/propose_query_simple.md"

    def __init__(self, question: str):
        placeholder_targets = {
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


class ProposeQueriesNoQuestions(Prompt):
    """Used to generate queries to answer AVeriTeC questions."""
    template_file_path = "defame/prompts/propose_queries_no_questions.md"

    def __init__(self, doc: Report):
        placeholder_targets = {
            "[DOC]": doc,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        queries = extract_queries(response)
        return dict(
            queries=queries,
            response=response,
        )


class AnswerCollectively(Prompt):
    """Used to generate answers to the AVeriTeC questions."""
    template_file_path = "defame/prompts/answer_question_collectively.md"

    def __init__(self, question: str, results: list[Source], doc: Report):
        result_strings = [f"## Result `{i}`\n{str(result)}" for i, result in enumerate(results)]
        results_str = "\n\n".join(result_strings)

        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULTS]": results_str,
        }
        super().__init__(placeholder_targets=placeholder_targets)

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
    template_file_path = "defame/prompts/answer_question.md"

    def __init__(self, question: str, result: Source, doc: Report):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
            "[RESULT]": result,
        }
        super().__init__(placeholder_targets=placeholder_targets)

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
    template_file_path = "defame/prompts/answer_question_no_evidence.md"

    def __init__(self, question: str, doc: Report):
        placeholder_targets = {
            "[DOC]": doc,
            "[QUESTION]": question,
        }
        super().__init__(placeholder_targets=placeholder_targets)


class DevelopPrompt(Prompt):
    template_file_path = "defame/prompts/develop.md"

    def __init__(self, doc: Report):
        placeholder_targets = {"[DOC]": doc}
        super().__init__(placeholder_targets=placeholder_targets)


class InterpretPrompt(Prompt):
    template_file_path = "defame/prompts/interpret.md"

    def __init__(self, content: Content, guidelines: str = None):
        placeholder_targets = {
            "[CONTENT]": content,
            "[GUIDELINES]": guidelines,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        paragraphs = response.split("\n")
        assert len(paragraphs) >= 2
        interpretation = paragraphs[0]
        topic = paragraphs[-1]
        return dict(
            interpretation=interpretation,
            topic=topic,
            response=response,
        )


class DecomposePrompt(Prompt):
    template_file_path = "defame/prompts/decompose.md"

    def __init__(self, content: Content):
        self.content = content
        placeholder_targets = {
            "[CONTENT]": content,
            "[INTERPRETATION]": content.interpretation
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        statements = response.split("\n\n")
        return dict(statements=[Claim(s.strip(), context=self.content) for s in statements if s],
                    response=response)


class JudgeNaively(Prompt):
    template_file_path = "defame/prompts/judge_naive.md"

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
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict:
        verdict = extract_verdict(response, classes=self.classes)
        return dict(verdict=verdict, response=response)


class JudgeMinimal(JudgeNaively):
    template_file_path = "defame/prompts/judge_minimal.md"


class InitializePrompt(Prompt):
    template_file_path = "defame/prompts/initialize.md"

    def __init__(self, claim: Claim):
        placeholder_targets = {
            "[CLAIM]": claim,
        }
        super().__init__(placeholder_targets=placeholder_targets)

class SM_RephrasePrompt(Prompt):
    template_file_path = "defame/prompts/sm_rephrase_input.md"

    def __init__(self, post: SocialMediaPost):
        placeholder_targets = {
            "[MESSAGE]": post.message,
        }
        super().__init__(placeholder_targets=placeholder_targets)
    # ggf noch custom extract methode

class SM_RetrieveClaim(Prompt):
    template_file_path = "defame/prompts/sm_retrieve_claim.md"

    def __init__(self, rephrased_msg: str, entities: str | list[str], post: SocialMediaPost):
        self.post = post

        if isinstance(entities, list):
            entities = ". ".join(entities)

        placeholder_targets = {
            "[PHRASES]": rephrased_msg,
            "[ENTITIES]": entities,
            "[IMAGES]": "No images available"
        }
        if post.has_images():
            placeholder_targets["[IMAGES]"] = self.post.images

        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> list[SocialMediaClaim]:
        claims = []
        pattern_claim = r"^-?\s*(.*?);\s*(.*?);\s*(.*)\s*$"
        pattern_image = r"<image:(\d+)>"
        for r in response.splitlines():
            # r should be of form "- entity1; relation1; entity2"
            m_text = re.match(pattern_claim, r)
            if m_text:
                head, relation, tail = m_text.groups()
                if head == "" or relation == "" or tail == "":
                    logger.warning(f"Skip claim {m_text} as it is not in triplet form")
                    continue
                m_img = re.match(pattern_image, tail)
                if m_img:
                    img_n = int(m_img.group(1))
                    if img_n <= len(self.post.images):
                        tail = self.post.images[img_n-1]
                    else:
                        logger.warning(f"Could not find image {tail}.")
                claims.append(SocialMediaClaim(posts={self.post}, head=head, relation=relation, tail=tail))
            else:
                logger.error(f"Could not extract claim from: {r}")

        return claims

class SM_CombineEntities(Prompt):
    template_file_path = "defame/prompts/sm_combine_entities.md"
    entities: set[str]
    def __init__(self, entities: set[str]):
        placeholder_targets = {
            "[ENTITIES]": entities,
        }
        super().__init__(placeholder_targets=placeholder_targets)


    def extract(self, response: str) -> dict:
            # return dict has entity to replace as key and new entity as value
            entities = [re.split(r",\s*|:\s*", g) for g in re.findall(r"\{([^}]*)}", response)]
            replacements = {}
            for group in entities:
                for entity in group:
                    if entity != group[0]:
                        replacements[entity] = group[0]

            return replacements

class SM_TextReport(Prompt):
    template_file_path = "defame/prompts/sm_summarize_text.md"

    def __init__(self, claims: list[SocialMediaClaim]):
        claims_formatted = []
        for c in claims:
            claims_formatted.append(f"({len(c.posts)}x) {str(c)}")

        placeholder_targets = {
            "[INPUT_CLAIMS]": claims_formatted,
        }

        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        return super().extract(response)

class SM_ImageReport(Prompt):
    template_file_path = "defame/prompts/sm_summarize_image.md"

    def __init__(self, claims: list[SocialMediaClaim], report:str):
        claims_formatted = []
        for c in claims:
            claims_formatted.extend([f"- ({len(c.posts)}x)", c.head, c.relation, c.tail])

        placeholder_targets = {
            "[INPUT_CLAIMS]": MultimodalSequence(*claims_formatted),
            "[INPUT_REPORT]": report,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> dict | str | None:
        return super().extract(response)


def load_exemplars(valid_actions: Collection[type[Action]]) -> str:
    exemplars_dir = Path("defame/prompts/plan_exemplars")
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
    from defame.evidence_retrieval.tools import ACTION_REGISTRY

    raw_action = raw_action.strip(" \"")

    if not raw_action:
        return None

    try:
        out = parse_function_call(raw_action)

        if out is None:
            raise ValueError(f'Invalid action: {raw_action}\nExpected format: action_name(<arg1>, <arg2>, ...)')

        action_name, args, kwargs = out

        for action in ACTION_REGISTRY:
            if action_name == action.name:
                return action(*args, **kwargs)

        raise ValueError(f'Invalid action: {raw_action}\nExpected format: action_name(<arg1>, <arg2>, ...)')

    except Exception as e:
        logger.warning(f"Failed to parse '{raw_action}':\n{e}")
        logger.warning(traceback.format_exc())

    return None


def extract_actions(answer: str, limit=5) -> list[Action]:
    from defame.evidence_retrieval.tools import ACTION_REGISTRY

    actions_str = extract_last_python_code_block(answer)

  # Handle cases where the LLM forgot to enclose actions in code block
    if not actions_str:
        candidates = []
        for action in ACTION_REGISTRY:
            pattern = re.compile(rf'({re.escape(action.name)}\(.+?\))', re.DOTALL)
            candidates += pattern.findall(answer)
        actions_str = "\n".join(candidates)
    if not actions_str:
        # Potentially prompt LLM to correct format: Expected format: action_name("arguments")
        return []

    # Parse actions
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
        label = Label(answer)
        assert label in classes
        return label

    except ValueError:
        # TODO: Verify if this is necessary
        # Maybe the label is a substring of the response
        for c in classes:
            if c.value in response:
                return c

    return None


def extract_queries(response: str) -> list:
    from defame.evidence_retrieval.tools import Search
    matches = find_code_span(response)
    queries = []
    for match in matches:
        query = strip_string(match)
        action = Search(f'"{query}"')
        queries.append(action)
    return queries


def extract_reasoning(answer: str) -> str:
    return remove_code_blocks(answer).strip()
