import re
from datetime import date
from typing import Any

import numpy as np
from jinja2.exceptions import TemplateSyntaxError
from openai import APIError

from infact.common import MultimediaSnippet, FCDocument, Model
from infact.prompts.prompts import SummarizeResultPrompt
from infact.tools.search.duckduckgo import DuckDuckGo
from infact.tools.search.knowledge_base import KnowledgeBase
from infact.tools.search.query_serper import SerperAPI
from infact.tools.search.search_api import SearchAPI
from infact.tools.search.wiki_dump import WikiDumpAPI
from infact.tools.tool import Tool
from infact.utils.console import gray, orange
from .common import SearchResult, Search, WebSearch, WikiDumpLookup
from ...common.misc import Query, WebSource

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
    summarize = True
    search_apis: dict[str, SearchAPI]
    stats: dict[str, int]

    def __init__(self,
                 search_engine_config: dict[str, dict] = None,
                 summarize: bool = True,
                 model: Model = None,
                 max_searches: int = 5,
                 limit_per_search: int = 5,
                 max_result_len: int = None,  # chars
                 do_debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        if search_engine_config is None:
            self.logger.log("No search engine specified. Falling back to DuckDuckGo.")
            search_engine_config = {"duckduckgo": {}}

        # Add device to knowledge base kwargs
        if "averitec_kb" in search_engine_config:
            search_engine_config["averitec_kb"].update(dict(device=self.device))

        # Setup search APIs
        self.search_apis = {se: SEARCH_APIS[se](logger=self.logger, **kwargs)
                            for se, kwargs in search_engine_config.items()}

        # Register available tools
        actions = []
        available_apis = self.search_apis.keys()
        if "wiki_dump" in available_apis:
            actions.append(WikiDumpLookup)
        if "google" in available_apis or "duckduckgo" in available_apis or "averitec_kb" in available_apis:
            actions.append(WebSearch)
        self.actions = actions

        self.model = model
        self.summarize = summarize
        self.max_searches = max_searches
        self.limit_per_search = limit_per_search
        self.max_result_len = max_result_len  # chars
        self.restrict_results_until_date = None

        self.debug = do_debug

        self.past_queries_helpful: dict[Query, bool] = {}
        # Cut the result text, maintaining a little buffer for the summary prompt
        self.past_search_results = set()

        self.reset()

    def _perform(self, action: Search) -> SearchResult:
        # Pick the strictest specified end date
        end_date = self.restrict_results_until_date
        if action.end_date is not None:
            if end_date is None or action.end_date < end_date:
                end_date = action.end_date

        # Prepare the query and run the search
        query = Query(text=action.query_string,
                      limit=self.limit_per_search,
                      start_date=action.start_date,
                      end_date=end_date)
        web_sources = self.search(query)

        return SearchResult(web_sources)

    def search(self, query: Query) -> list[WebSource]:
        """Searches for evidence using the search APIs according to their precedence."""

        for search_engine in list(self.search_apis.values()):
            results = self._retrieve_search_results(query, search_engine)

            # Track search engine call
            self.stats[search_engine.name] += 1

            # Log search results info
            self.logger.log(f"Got {len(results)} new result(s):")
            for i, result in enumerate(results):
                result_summary = f"\t{i + 1}."
                if result.date is not None:
                    result_summary += f" {result.date.strftime('%B %d, %Y')}"
                result_summary += f" {result.url}"
                self.logger.log(result_summary)

            # Modify the results text to avoid jinja errors when used in prompt
            results = self._postprocess_results(results)

            # If there is at least one result, we were successful
            if len(results) > 0:
                self._register_search_results(results)
                return results
        return []

    def _remove_known_search_results(self, results: list[WebSource]) -> list[WebSource]:
        """Removes already known search results"""
        return [r for r in results if r not in self.past_search_results]

    def _register_search_results(self, results: list[WebSource]):
        """Adds the provided list of results to the set of known results."""
        self.past_search_results |= set(results)

    def reset(self):
        """Removes all known search results and resets the statistics."""
        self.past_search_results = set()
        self.stats = {s.name: 0 for s in self.search_apis.values()}

    def _retrieve_search_results(
            self,
            query: Query,
            search_engine: SearchAPI,
    ) -> list[WebSource]:
        # Run the search
        results = search_engine.search(query)
        self.past_queries_helpful[query] = True

        # Remove already known results
        return self._remove_known_search_results(results)

    def _postprocess_results(self, results: list[WebSource]) -> list[WebSource]:
        """Modifies the results text to avoid jinja errors when used in prompt."""
        for result in results:
            result.text = self.postprocess_result(result.text)
        return results

    def postprocess_result(self, result: str):
        """Removes all double curly braces to avoid conflicts with Jinja and optionally truncates
        the result text to a maximum length."""
        result = re.sub(r"\{\{.*}}", "", result)
        if self.max_result_len is not None:
            result = result[self.max_result_len:]
        return result

    def _summarize(self, result: SearchResult, **kwargs) -> MultimediaSnippet:
        doc = kwargs.get("doc")
        for web_source in result.sources:
            self._summarize_single_web_source(web_source, doc)
        # TODO: Implement summary of summaries
        summary = "\n\n".join(map(str, result.sources))
        return MultimediaSnippet(summary)

    def _summarize_single_web_source(self, web_source: WebSource, doc: FCDocument):
        prompt = SummarizeResultPrompt(web_source, doc)

        try:
            summary = self.model.generate(prompt, max_attempts=3)
        except APIError as e:
            self.logger.log(orange(f"APIError: {e} - Skipping the summary for {web_source.url}."))
            self.logger.log(orange(f"Used prompt:\n{str(prompt)}"))
            summary = "NONE"
        except TemplateSyntaxError as e:
            self.logger.log(orange(f"TemplateSyntaxError: {e} - Skipping the summary for {web_source.url}."))
            summary = "NONE"
        except ValueError as e:
            self.logger.log(orange(f"ValueError: {e} - Skipping the summary for {web_source.url}."))
            summary = "NONE"
        except Exception as e:
            self.logger.log(orange(f"Error while summarizing! {e} - Skipping the summary for {web_source.url}."))
            summary = "NONE"

        web_source.summary = MultimediaSnippet(summary)

        if web_source.is_useful():
            self.logger.log("Useful result: " + gray(str(web_source)))

    def get_stats(self) -> dict[str, Any]:
        total_searches = np.sum([n for n in self.stats.values()])
        return {
            "Total searches": total_searches,
            "Search engine calls": self.stats,
        }

    def set_date_restriction(self, until: date):
        self.restrict_results_until_date = until
