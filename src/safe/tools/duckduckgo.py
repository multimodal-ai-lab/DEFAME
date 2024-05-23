from duckduckgo_search import DDGS
from typing import List, Dict, Any
from common.console import red, bold

class DuckDuckGo:
    """Class for querying the DuckDuckGo API."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def run(self, query: str) -> Dict[str, Any]:
        """Run a search query and return structured results."""
        results = DDGS().text(query, max_results=self.max_results)
        if not results:
            print(bold(red("DuckDuckGo is having issues. Run the duckduckgo.py and check https://duckduckgo.com/ for more information.")))
        return self._parse_results(results)

    def _parse_results(self, results: List[Dict[str, str]]) -> str:
        """Parse results from DuckDuckGo search and return string with each line: title: body"""
        snippets = []
        for result in results:
            snippets.append(f'{result.get("title", "")}: {result.get("body", "")}.')
        return '\n'.join(snippets)


if __name__ == "__main__":
    duckduckgo_api = DuckDuckGo(max_results=5)
    
    query = "Sean Connery letter Steve Jobs"
    results = duckduckgo_api.run(query)
    
    print("Search Results:", results)



