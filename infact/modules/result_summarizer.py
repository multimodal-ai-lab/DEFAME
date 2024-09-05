from jinja2.exceptions import TemplateSyntaxError
from openai import APIError

from infact.common.document import FCDocument
from infact.common.modeling import Model
from infact.common.results import Result, SearchResult
from infact.common.logger import Logger
from infact.prompts.prompt import SummarizeResultPrompt, SelectionPrompt
from infact.utils.console import gray, orange, num2text
from infact.utils.parsing import extract_answer_and_url


class ResultSummarizer:
    """Summarizes any collection of (search etc.) results w.r.t. the current
    fact-checking document."""

    def __init__(self, model: Model, logger: Logger):
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

                try:
                    result.summary = self.model.generate(prompt, max_attempts=3)
                except APIError as e:
                    self.logger.log(orange(f"APIError: {e} - Skipping the summary for {result.url}."))
                    self.logger.log(orange(f"Used prompt:\n{str(prompt)}"))
                    result.summary = "NONE"
                except TemplateSyntaxError as e:
                    self.logger.log(orange(f"TemplateSyntaxError: {e} - Skipping the summary for {result.url}."))
                    result.summary = "NONE"
                except ValueError as e:
                    self.logger.log(orange(f"ValueError: {e} - Skipping the summary for {result.url}."))
                    result.summary = "NONE"
                except Exception as e:
                    self.logger.log(orange(f"Error while summarizing! {e} - Skipping the summary for {result.url}."))
                    result.summary = "NONE"

                if result.is_useful():
                    self.logger.log("Useful result: " + gray(str(result)))
            else:
                result.summary = str(result)

        return results
