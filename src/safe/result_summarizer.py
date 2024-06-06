from common.document import FCDoc
from common.modeling import Model
from common.results import Result, SearchResult
from eval.logger import EvaluationLogger
from safe.prompts.prompt import SummarizeResultPrompt
from common.console import gray


class ResultSummarizer:
    """Summarizes any collection of (search etc.) results w.r.t. the current
    fact-checking document."""

    def __init__(self, model: Model, logger: EvaluationLogger):
        self.model = model
        self.logger = logger

    def summarize(self, results: list[Result], doc: FCDoc) -> set[Result]:
        """Summarizes each result in results and adds the summary to each result."""
        results = set(results)
        self.logger.log(f"Summarizing {len(results)} unique result(s)...")
        for result in results:
            if isinstance(result, SearchResult):
                summarize_result_prompt = SummarizeResultPrompt(result.text, doc)
                result.summary = self.model.generate(str(summarize_result_prompt))
                if result.is_useful():
                    self.logger.log(gray(str(result)))
            else:
                raise NotImplementedError()
        return results
