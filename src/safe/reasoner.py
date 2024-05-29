import dataclasses
import re
from typing import Sequence, Optional

from common import utils
from common.console import red
from common.label import Label
from common.modeling import Model
from eval.logger import EvaluationLogger
from safe.config import debug_safe, max_steps, max_retries
from safe.prompts.prompt import ReasonPrompt
from safe.searcher import SearchResult
from eval.benchmark import Benchmark


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class Reasoner:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self, model: Model, benchmark: Benchmark):
        self.model = model
        self.benchmark = benchmark
        self.debug = debug_safe
        self.max_steps = max_steps
        self.max_retries = max_retries

    def reason(self,
               claim: str,
               evidence: Sequence[SearchResult],
               logger: Optional[EvaluationLogger] = None
    ) -> (Label, str):
        """Takes the claim and the gathered evidence, determines the
        claim's veracity through reasoning and returns the verdict with
        the reasoning as justification."""
        final_answer, num_tries = None, 0
        while not final_answer and num_tries <= self.max_retries:
            num_tries += 1
            final_answer = self.maybe_get_final_answer(claim, evidence=evidence, logger=logger)

        if final_answer is None:
            utils.maybe_print_error('Unsuccessful parsing for `final_answer`')
            if logger is not None:
                logger.log('Unsuccessful parsing for `final_answer`')

        predicted_label = Label(final_answer.answer.lower())

        return predicted_label, final_answer.response

    def maybe_get_final_answer(self,
                               claim: str,
                               evidence: Sequence[SearchResult],
                               verbose: bool = False,
                               logger: Optional[EvaluationLogger] = None,
                               ) -> FinalAnswer | None:
        """Get the final answer from the model."""
        knowledge = '\n'.join([search.result for search in evidence if search.result is not None])
        reason_prompt = ReasonPrompt(claim, knowledge, self.benchmark)
        model_response = self.model.generate(str(reason_prompt), do_debug=self.debug)
        if model_response.startswith("I cannot") or model_response.startswith("I'm sorry"):
            utils.print_guard()
            if logger is not None:
                logger.log('f"Hit the guard rail with prompt: {claim}"')
            answer = 'refused'
            return FinalAnswer(response=model_response, answer=answer)
        answer = utils.extract_first_square_brackets(model_response)
        answer = re.sub(r'[^\w\s]', '', answer).strip()

        valid_labels = [label.value.lower() for label in Label]
        if model_response and answer.lower() in valid_labels:
            return FinalAnswer(response=model_response, answer=answer)
        else:
            # Adjust the model response
            select = f"Respond with one word! From {valid_labels}, select the most fitting for the following string:\n"
            adjusted_response = self.model.generate(select + model_response).lower()
            if verbose:
                utils.print_wrong_answer(model_response, adjusted_response)
            if logger:
                logger.log(f"No answer label was found - likely due to wrong formatting.Model Output: {model_response}")
                logger.log(f"Adjusted Output: {adjusted_response}")
            if adjusted_response not in valid_labels:
                print(red(f"Error in generating answer. Defaulting to '{Label.REFUSED_TO_ANSWER}'\n"))
                if logger is not None:
                    logger.log(f"Error in generating answer. Defaulting to '{Label.REFUSED_TO_ANSWER}'")
                return FinalAnswer(response=model_response, answer=Label.REFUSED_TO_ANSWER.value)
            else:
                return FinalAnswer(response=model_response, answer=adjusted_response)
