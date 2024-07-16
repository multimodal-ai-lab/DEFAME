import re

from src.common.action import WebSearch, WikiDumpLookup, Search
from src.common.results import SearchResult
from src.eval.logger import EvaluationLogger
from src.tools.search.duckduckgo import DuckDuckGo
from src.tools.search.knowledge_base import KnowledgeBase
from src.tools.search.query_serper import SerperAPI
from src.tools.search.search_api import SearchAPI
from src.tools.search.wiki_dump import WikiDumpAPI
from src.tools.tool import Tool

SEARCH_APIS = {
    "google": SerperAPI,
    "duckduckgo": DuckDuckGo,
    "wiki_dump": WikiDumpAPI,
    "averitec_kb": KnowledgeBase,
}


class Searcher(Tool):
    """Searches the specified resource (Google, Wikipedia, ...) for evidence. Takes
    a list of specified search engines. The list defines the precedence of the search
    engines meaning that if search_engine[0] did not yield any results, search_engine[1]
    will be tried next."""
    # TODO: Rank or annotate the websites according to their credibility, like MUSE
    name = "searcher"
    search_apis: dict[str, SearchAPI]

    def __init__(self,
                 search_engines: list[str] = None,
                 logger: EvaluationLogger = None,
                 summarize: bool = True,
                 max_searches: int = 5,
                 limit_per_search: int = 5,
                 do_debug: bool = False):
        self.logger = logger or EvaluationLogger()

        if search_engines is None:
            self.logger.log("No search engine specified. Falling back to DuckDuckGo.")
            search_engines = ["duckduckgo"]

        # Setup search APIs
        self.search_apis = {se: SEARCH_APIS[se](logger=self.logger) for se in search_engines}

        # Register available tools
        actions = []
        if "wiki_dump" in search_engines:
            actions.append(WikiDumpLookup)
        if "google" in search_engines or "duckduckgo" in search_engines or "averitec_kb" in search_engines:
            actions.append(WebSearch)
        self.actions = actions

        self.summarize = summarize
        self.max_searches = max_searches
        self.limit_per_search = limit_per_search

        self.debug = do_debug

        self.past_queries_helpful: dict[str, bool] = {}
        # Cut the result text, maintaining a little buffer for the summary prompt
        self.past_search_results = set()

    def perform(self, action: Search) -> list[SearchResult]:
        return self.search(action.query)

    def search(self, query: str) -> list[SearchResult]:
        """Searches for evidence using the search APIs according to their precedence."""
        for search_engine in list(self.search_apis.values()):
            results = self._retrieve_search_results(query, search_engine)

            # Log search results info
            self.logger.log(f"Got {len(results)} new result(s):")
            for i, result in enumerate(results):
                self.logger.log(f"\t{i + 1}. {result.source}")

            # Modify the results text to avoid jinja errors when used in prompt
            results = self._postprocess_results(results)

            # If there is at least one result, we were successful
            if len(results) > 0:
                self._register_search_results(results)
                return results
        return []

    def _remove_known_search_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Removes already known search results"""
        return [r for r in results if r not in self.past_search_results]

    def _register_search_results(self, results: list[SearchResult]):
        """Adds the provided list of results to the set of known results."""
        self.past_search_results |= set(results)

    def reset(self):
        """Removes all known search results."""
        self.past_search_results = set()

    def _retrieve_search_results(
            self,
            query: str,
            search_engine: SearchAPI,
    ) -> list[SearchResult]:
        # Run the search
        results = search_engine.search(query, self.limit_per_search)
        self.past_queries_helpful[query] = True

        # Remove already known results
        return self._remove_known_search_results(results)

    def _postprocess_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Modifies the results text to avoid jinja errors when used in prompt."""
        for result in results:
            result.text = postprocess_result(result.text)
        return results


def postprocess_result(result: str):
    """Removes all double curly braces to avoid conflicts with Jinja."""
    result = re.sub(r"\{\{.*}}", "", result)
    return result
