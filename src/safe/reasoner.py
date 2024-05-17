import dataclasses
import re

from common import utils
from common.label import Label
from common.modeling import Model
from safe.config import debug_safe, max_steps, max_retries
from safe.prompts.common import STATEMENT_PLACEHOLDER, KNOWLEDGE_PLACEHOLDER
from safe.prompts.reasoning import FINAL_ANSWER_FORMAT
from safe.searcher import SearchResult


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class Reasoner:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self, model: Model):
        self.model = model

        self.debug = debug_safe
        self.max_steps = max_steps
        self.max_retries = max_retries

    def reason(self, claim: str, evidence: list[SearchResult]) -> (Label, str):
        """Takes the claim and the gathered evidence, determines the
        claim's veracity through reasoning and returns the verdict with
        the reasoning as justification."""
        final_answer, num_tries = None, 0
        while not final_answer and num_tries <= self.max_retries:
            num_tries += 1
            final_answer = self.maybe_get_final_answer(claim, evidence=evidence)

        if final_answer is None:
            utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

        predicted_label = Label(final_answer.answer)

        return predicted_label, final_answer.response

    def maybe_get_final_answer(self,
                               claim: str,
                               evidence: list[SearchResult],
                               ) -> FinalAnswer | None:
        """Get the final answer from the model."""
        knowledge = '\n'.join([search.result for search in evidence])
        full_prompt = FINAL_ANSWER_FORMAT.replace(STATEMENT_PLACEHOLDER, claim)
        full_prompt = full_prompt.replace(KNOWLEDGE_PLACEHOLDER, knowledge)
        full_prompt = utils.strip_string(full_prompt)
        model_response = self.model.generate(full_prompt, do_debug=self.debug)
        if model_response.startswith("I cannot"):
            print("Model hit the railguards -.-'. Defaulting to NOT_SUPPORTED.")
            model_response = '[NOT_SUPPORTED_LABEL]'
        answer = utils.extract_first_square_brackets(model_response)
        answer = re.sub(r'[^\w\s]', '', answer).strip()

        valid_labels = [label.value for label in Label]
        if model_response and answer in valid_labels:
            return FinalAnswer(response=model_response, answer=answer)
