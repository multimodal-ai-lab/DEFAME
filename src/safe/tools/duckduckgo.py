import time
from typing import List, Dict

from duckduckgo_search import DDGS

from common.console import red, bold
from eval.logger import EvaluationLogger


class DuckDuckGo:
    """Class for querying the DuckDuckGo API."""

    def __init__(self,
                 max_results: int = 5,
                 max_retries: int = 10,
                 backoff_factor: float = 60.0,
                 logger: EvaluationLogger = None):
        self.max_results = max_results
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = EvaluationLogger() if logger is None else logger

    def run(self, query: str) -> str:
        """Run a search query and return structured results."""
        attempt = 0
        while attempt < self.max_retries:
            if attempt > 3:
                wait_time = self.backoff_factor * (attempt)
                self.logger.log(bold(red(f"Sleeping {wait_time} seconds.")))
                time.sleep(wait_time)
            try:
                results = DDGS().text(query, max_results=self.max_results)
                if not results:
                    self.logger.log(bold(red("DuckDuckGo is having issues. Run duckduckgo.py "
                                             "and check https://duckduckgo.com/ for more information.")))
                    return ''
                return self._parse_results(results)
            except Exception as e:
                attempt += 1
                query += '?'
                self.logger.log((bold(red(f"Attempt {attempt} failed: {e}. Retrying with modified query..."))))
        self.logger.log(bold(red("All attempts to reach DuckDuckGo have failed. Please try again later.")))

        return ''

    def _parse_results(self, results: List[Dict[str, str]]) -> str:
        """Parse results from DuckDuckGo search and return structured dictionary."""
        snippets = []
        for result in results:
            snippets.append(f'{result.get("title", "")}: {result.get("body", "")}.')
        return '\n'.join(snippets)


if __name__ == "__main__":
    duckduckgo_api = DuckDuckGo(max_results=5)

    query = "Sean Connery letter Steve Jobs"
    results = duckduckgo_api.run(query)

    print("Search Results:", results)
