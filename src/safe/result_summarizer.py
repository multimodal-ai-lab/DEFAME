from common.document import FCDocument
from common.modeling import Model
from common.results import Result, SearchResult
from eval.logger import EvaluationLogger
from safe.prompts.prompt import SummarizeResultPrompt
from common.console import gray, orange
from openai.error import InvalidRequestError


class ResultSummarizer:
    """Summarizes any collection of (search etc.) results w.r.t. the current
    fact-checking document."""

    def __init__(self, model: Model, logger: EvaluationLogger):
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
                summarize_result_prompt = SummarizeResultPrompt(result, doc)
                try:
                    result.summary = self.model.generate(str(summarize_result_prompt),
                                                         max_attempts=10)
                except InvalidRequestError as e:
                    self.logger.log(orange(f"InvalidRequestError: {e} - Skipping this summary"))
                    self.logger.log(orange(f"Used prompt:\n{str(summarize_result_prompt)}"))
                    result.summary = "NONE"
                if result.is_useful():
                    self.logger.log("Useful result: " + gray(str(result)))
            else:
                raise NotImplementedError()
        return results
