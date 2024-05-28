import time
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
import logging
from logging import Logger
from common.console import red, bold
from eval.logging import print_log

logging.getLogger('duckduckgo_search').setLevel(logging.WARNING)

class DuckDuckGo:
    """Class for querying the DuckDuckGo API."""

    def __init__(self, max_results: int = 5, max_retries: int = 10, backoff_factor: float = 60.0):
        self.max_results = max_results
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def run(self, query: str, verbose: bool = False, logger: Optional[Logger] = None) -> str:
        """Run a search query and return structured results."""
        attempt = 0
        while attempt < self.max_retries:
            if attempt > 3:
                wait_time = self.backoff_factor * (attempt)
                if verbose:
                    print(bold(red(f"Sleeping {wait_time} seconds.")))
                if logger:
                    print_log(logger, f"Sleeping {wait_time} seconds.")
                time.sleep(wait_time)
            try:
                results = DDGS().text(query, max_results=self.max_results)
                if not results:
                    print(bold(red("DuckDuckGo is having issues. Run the duckduckgo.py and check https://duckduckgo.com/ for more information.")))
                    return ''
                return self._parse_results(results)
            except Exception as e:
                attempt += 1
                query += '?'
                if verbose:
                    print(bold(red(f"Attempt {attempt} failed: {e}. Retrying with modified query...")))
                if logger:
                    print_log(logger, f"Attempt {attempt} failed: {e}. Retrying with modified query: {query}")
        if verbose:
            print(bold(red("All attempts to reach DuckDuckGo have failed. Please try again later.")))
        if logger:
            print_log(logger, f"All attempts to reach DuckDuckGo have failed.")        
        
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
