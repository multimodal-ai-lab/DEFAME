import re
from typing import Optional

import numpy as np

from common.console import gray, orange, num2text
from common.modeling import Model
from common.utils import GUARDRAIL_WARNING, extract_first_code_block
from eval.logger import EvaluationLogger
from safe.config import debug_safe, num_searches
from safe.prompts.prompt import SearchPrompt, SummarizePrompt
from safe.tools.search.duckduckgo import DuckDuckGo
from safe.tools.search.knowledge_base import KnowledgeBase
from safe.tools.search.query_serper import SerperAPI
from safe.tools.search.search_api import SearchAPI
from common.results import SearchResult
from safe.tools.search.wiki_dump import WikiDumpAPI

SEARCH_APIS = {
    "google": SerperAPI,
    "duckduckgo": DuckDuckGo,
    "wiki_dump": WikiDumpAPI,
    "averitec_kb": KnowledgeBase,
}


class Searcher:
    """Searches the specified resource (Google, Wikipedia, ...) for evidence. Takes
    a list of specified search engines. The list defines the precedence of the search
    engines meaning that if search_engine[0] did not yield any results, search_engine[1]
    will be tried next."""
    # TODO: Rank or annotate the websites according to their credibility, like MUSE
    search_apis: dict[str, SearchAPI]

    def __init__(self, search_engines: list[str],
                 model: Model,
                 logger: EvaluationLogger = None,
                 summarize: bool = True,
                 max_searches: int = 5,
                 limit_per_search: int = 5):
        self.logger = logger or EvaluationLogger()
        self.search_apis = {se: SEARCH_APIS[se](logger=self.logger) for se in search_engines}
        self.model = model

        self.summarize = summarize
        self.max_searches = max_searches
        self.limit_per_search = limit_per_search

        self.debug = debug_safe

        self.past_queries_helpful: dict[str, bool] = {}
        # Cut the result text, maintaining a little buffer for the summary prompt
        self.past_search_results = set()

    def search(self, query: str) -> list[SearchResult]:
        """Searches for evidence using the search APIs according to their precedence."""
        for search_engine in list(self.search_apis.values()):
            results = self._retrieve_search_results(query, search_engine)

            # Log search results info
            self.logger.log(f"Got {len(results)} new result(s):")
            for i, result in enumerate(results):
                self.logger.log(f"\t{i+1}. {result.source}")

            # Modify the results text to avoid jinja errors when used in prompt
            results = self._postprocess_results(results)

            # If there is at least one result, we were successful
            if len(results) > 0:
                self._add_known_search_results(results)
                return results
        return []

    def _remove_known_search_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Removes already known search results"""
        return [r for r in results if r not in self.past_search_results]

    def _add_known_search_results(self, results: list[SearchResult]):
        """Adds the provided list of results to the set of known results."""
        self.past_search_results |= set(results)

    def reset(self):
        """Removes all known search results."""
        self.past_search_results = set()

    def find_evidence(
            self,
            claim: str,
    ) -> list[SearchResult]:
        """Main method of the Searcher class implementing the searching strategy.
        Takes a claim, produces proper search queries to find sufficient evidence that helps to
        verify the claim and returns the retrieved list of (summarized) search results."""

        # Start with a first ad-hoc search by simply using the original claim as the search query
        first_precedence_search_engine = list(self.search_apis.values())[0]
        search_results = self._retrieve_search_results(claim, first_precedence_search_engine, claim)

        # As long as the information is insufficient for a conclusive veracity prediction,
        # continue gathering more information
        n_searches = 0
        while not self.sufficient_knowledge(claim, search_results) and n_searches < self.max_searches:
            new_results = self._gather_more_results(claim, past_results=search_results)
            search_results.extend(new_results)
            n_searches += 1

        self.past_queries_helpful = {}  # reset
        return search_results

    def _gather_more_results(self,
                             claim: str,
                             past_results: list[SearchResult],
                             ) -> list[SearchResult]:
        """Generates a search query and runs it to find new information that helps
        to verify the claim and is not already contained in past_results. Tries a
        search engine with lower precedence if the one with higher precedence didn't
        yield and results."""
        # Try the search engines according to their precedence
        for search_engine in list(self.search_apis.values()):
            query = self._generate_query(claim, search_engine.name, past_results)
            results = self._retrieve_search_results(query, search_engine, claim, past_results)

            # Register the used query and save its usefulness
            self.past_queries_helpful[query] = np.any([result.is_useful() for result in results])

            # If there is at least one result, we were successful TODO: restrict to useful results
            if len(results) > 0:
                return results

        return []

    def _generate_query(self,
                        claim: str,
                        search_engine_name: str,
                        past_results: list[SearchResult]):
        """Produces the next query depending on the search engine to use. Ensures that
        the query is different to all previous queries."""
        # Construct the prompt tasking the model to produce a search query
        knowledge = extract_knowledge(past_results)
        past_queries = [query if self.past_queries_helpful[query] else "Bad query: " + query
                        for query in list(self.past_queries_helpful.keys())]
        past_queries_str = '\n'.join(past_queries) or 'N/A'

        # Get the query and avoid it to be the same as any of the past queries
        query = None
        while query is None or query in past_queries:
            search_prompt = SearchPrompt(claim, knowledge, past_queries_str,
                                         search_engine=search_engine_name,
                                         open_source=self.model.open_source)
            model_response = self.model.generate(str(search_prompt), do_debug=self.debug)
            query = self._extract_query(model_response)
            if query in past_queries:
                self.logger.log(orange("Duplicate query proposed. Trying again..."))
                # Change the prompt to enable different answer for deterministic model
                past_queries_str += f"\nBad query: {query}"

        return query

    def _extract_query(self, model_response: str) -> Optional[str]:
        model_response = model_response.replace('"', '')

        # Check if the LLM safeguard was hit
        if model_response.startswith("I cannot") or model_response.startswith("I'm sorry"):
            self.logger.log(GUARDRAIL_WARNING)
            return None

        query = extract_first_code_block(model_response, ignore_language=True)
        if not query:
            query = self._post_process_query(model_response)

        return query

    def _post_process_query(
            self,
            model_response: str,
    ) -> str:
        """
        Processes the model response to extract the query. Ensures correct formatting
        and adjusts the response if needed.
        """

        # If query extraction was unsuccessful, use the LLM to extract the query from the response
        self.logger.log(f"No query was found in output - likely due to "
                        f"wrong formatting.\nModel Output: {model_response}")

        instruction = "Extract a simple sentence that I can use for a Google Search query from this string:\n"
        query = self.model.generate(instruction + model_response)

        # Remove unwanted newlines
        query = query.replace('\n', '')
        re.sub(r'[\n`´]', '', query)

        return query

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

    def _summarize_result(self, result: SearchResult, claim: str):
        summarize_prompt = SummarizePrompt(claim, result.query, result.text)
        result.summary = self.model.generate(str(summarize_prompt), do_debug=self.debug)
        self.logger.log(f"Summarized result: " + gray(result.summary))

    def sufficient_knowledge(
            self,
            claim: str,
            past_results: list[SearchResult],
    ) -> bool:
        """
        # TODO: Replace this by the standard reasoner
        This function uses an LLM to evaluate the sufficiency of search_results.
        """
        knowledge = extract_knowledge(past_results)
        instruction = ("Given the following INFORMATION, determine if it is enough to conclusively decide "
                       "whether the Claim is true or false with high certainty. If the INFORMATION is sufficient, "
                       "respond 'sufficient'. Otherwise, respond 'insufficient'. "
                       "If you are in doubt or need more information, respond 'insufficient'. "
                       "Respond with only one word.")
        input = f"{instruction}\INFORMATION:\n{knowledge}\Claim:{claim}"
        model_decision = self.model.generate(input)
        if model_decision.lower() == "sufficient":
            self.logger.log(f"Sufficient knowledge found.")
            return True
        else:
            return False


def postprocess_result(result: str):
    """Removes all double curly braces to avoid conflicts with Jinja."""
    result = re.sub(r"\{\{.*}}", "", result)
    return result


def extract_knowledge(search_results: list[SearchResult]) -> str:
    knowledge = [result.summary for result in search_results if result.is_useful()]
    return '\n'.join(knowledge) or 'N/A'
