from jinja2.exceptions import TemplateSyntaxError
from openai.error import InvalidRequestError

from src.common.document import FCDocument
from src.common.modeling import LLM
from src.common.results import Result, SearchResult
from src.eval.logger import EvaluationLogger
from src.prompts.prompt import SummarizeResultPrompt, SelectionPrompt
from src.utils.console import gray, orange, num2text
from src.utils.parsing import extract_answer_and_url


class ResultSummarizer:
    """Summarizes any collection of (search etc.) results w.r.t. the current
    fact-checking document."""

    def __init__(self, model: LLM, logger: EvaluationLogger):
        self.model = model
        self.logger = logger

    def summarize(self, results: list[Result], doc: FCDocument) -> set[Result]:
        """Summarizes each result in results and adds the summary to each result."""
        results = set(results)
        if len(results) == 0:
            return results

        self.logger.log(f"Summarizing {len(results)} unique result(s)...")
        for result in results:
            if isinstance(result, SearchResult):
                prompt = SummarizeResultPrompt(result, doc)
                prompt = self._maybe_truncate_prompt(str(prompt))

                try:
                    result.summary = self.model.generate(prompt, max_attempts=3)
                except InvalidRequestError as e:
                    self.logger.log(orange(f"InvalidRequestError: {e} - Skipping the summary for {result.source}."))
                    self.logger.log(orange(f"Used prompt:\n{str(prompt)}"))
                    result.summary = "NONE"
                except TemplateSyntaxError as e:
                    self.logger.log(orange(f"TemplateSyntaxError: {e} - Skipping the summary for {result.source}."))
                    result.summary = "NONE"
                except ValueError as e:
                    self.logger.log(orange(f"ValueError: {e} - Skipping the summary for {result.source}."))
                    result.summary = "NONE"
                except Exception as e:
                    self.logger.log(orange(f"Error while summarizing! {e} - Skipping the summary for {result.source}."))
                    result.summary = "NONE"

                if result.is_useful():
                    self.logger.log("Useful result: " + gray(str(result)))
            else:
                result.summary = str(result)

        return results

    def _maybe_truncate_prompt(self, prompt: str) -> str:
        """Truncates the prompt if it's too long (exceeds the LLM context length limit)."""
        num_prompt_tokens = self.model.count_tokens(prompt)
        if num_prompt_tokens > self.model.max_prompt_len:
            self.logger.log(orange(f"INFO: Truncating search result due to excess length. Cutting away "
                                   f"{num2text(num_prompt_tokens - self.model.max_prompt_len)} tokens to fit into "
                                   f"LLM context window of {num2text(self.model.context_window)} tokens."))
            return prompt[:self.model.max_prompt_len * 3]  # count conservatively with only 3 chars per token
        else:
            return prompt

    def _extract_most_fitting(self, question, results: list[Result]) -> tuple[str, str]:
        n_results = len(results) or 1
        result_len_limit = self.model.max_prompt_len / n_results
        results_truncated = []
        for result in results:
            result_len = self.model.count_tokens(result.text)
            if result_len > result_len_limit:
                # self.logger.log(orange(f"FACTUALITY QUESTION INFO: Truncating search result due to excess length. Cutting away "
                #                   f"{num2text(result_len - result_len_limit)} tokens to fit into "
                #                   f"LLM context window of {num2text(self.model.context_window)} tokens."))
                result.text = result.text[:int(result_len_limit * 3)]
            results_truncated.append(result)
        prompt = SelectionPrompt(question, results_truncated)
        generated_result = self.model.generate(str(prompt), max_attempts=3)
        generated_answer, url = extract_answer_and_url(generated_result)

        return generated_answer, url
