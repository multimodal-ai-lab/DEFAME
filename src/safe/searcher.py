import dataclasses
<<<<<<< HEAD
from typing import Optional, Sequence, List
=======
from typing import Optional, Sequence
import re
>>>>>>> 3b0cd46f8decb5c6faffa5236491aa300adf723e

from common import utils
from common.modeling import Model
from common.shared_config import serper_api_key
from safe.config import num_searches, debug_safe, max_steps, max_retries
from safe.prompts.prompt import SearchPrompt, SummarizePrompt
from safe.tools.query_serper import SerperAPI
from safe.tools.wiki_dump import WikiDumpAPI


@dataclasses.dataclass()
class SearchResult:
    query: str
    result: str

    def __str__(self):
        return f"SearchResult(\n\tquery='{self.query}'\n\tresult='{self.result}'\n)"


class Searcher:
    """Searches the specified resource (Google, Wikipedia, ...) for evidence."""

    def __init__(self, search_engine: str, model: Model):
        assert search_engine in ["google", "wiki"]
        self.search_engine = search_engine
        self.model = model

        self.serper_searcher = SerperAPI(serper_api_key, k=num_searches)
        self.wiki_searcher = WikiDumpAPI()

        self.max_steps = max_steps
        self.max_retries = max_retries
        self.debug = debug_safe

    def search(self, claim,limit_search=True, verbose: bool = False) -> Sequence[SearchResult]:
        search_results = []

        for _ in range(self.max_steps):
            next_search, num_tries = None, 0

            while not next_search and num_tries <= self.max_retries:
                next_search = self._maybe_get_next_search(claim, search_results, verbose=verbose)
                num_tries += 1

            if next_search is None:
                utils.maybe_print_error('Unsuccessful parsing for `next_search`')
                break
            else:
                search_results.append(next_search)

            if limit_search and self.sufficient_knowledge(claim, search_results):
                if verbose:
                    print("LLM decided that the current knowledge is sufficient.")
                break

        return search_results

    def _maybe_get_next_search(self,
                               claim: str,
                               past_searches: list[SearchResult],
                               verbose: Optional[bool] = False,
                               ) -> SearchResult | None:
        """Get the next query from the model, use the query to search for evidence and return it."""
        # Construct the prompt tasking the model to produce a search query
        past_results = [s.result for s in past_searches if s.result is not None]
        knowledge = '\n'.join(past_results)
        knowledge = 'N/A' if not knowledge else knowledge
        past_queries = [s.query for s in past_searches]
        past_queries_str = '\n'.join(past_queries)
        past_queries_str = 'N/A' if not past_queries_str else past_queries_str
        search_prompt = SearchPrompt(claim, knowledge, past_queries_str,
                                     search_engine=self.search_engine,
                                     open_source=self.model.open_source)

        # Get and validate the model's response
        model_response = self.model.generate(str(search_prompt), do_debug=self.debug).replace('"', '')
        if model_response.startswith("I cannot") or model_response.startswith("I'm sorry"):
            if verbose:
                utils.print_guard()
            model_response = claim
        query = utils.extract_first_code_block(model_response, ignore_language=True)
        if not query:
            query = self.post_process_query(model_response)

        # Avoid casting the same, previously used query again
        if query in past_queries:
            return

        result = self._call_api(query)

        # Avoid duplicate results
        if result in past_results:
            result = None  # But keep query to avoid future duplicates

        # If result is too long, summarize it (to avoid hitting the context length limit)
        if result is not None and len(result) > 512:
            print("Summarizing result:", result)
            summarize_prompt = SummarizePrompt(query, result)
            result = self.model.generate(str(summarize_prompt), do_debug=self.debug)

        search_result = SearchResult(query=query, result=result)

        if verbose:
            print("Found", search_result)

        return search_result

    def post_process_query(self, model_response: str, verbose: bool = False) -> str:
        """
        Processes the model response to extract the query. Ensures correct formatting
        and adjusts the response if needed.
        """

        if verbose and model_response.startswith("I cannot"):
            utils.print_guard()

        query = utils.extract_first_code_block(model_response, ignore_language=True)

        # If query extraction was unsuccessful, use the LLM to extract the query from the response
        if not query:
            if verbose:
                print(f"No query was found in output - likely due to wrong formatting.\nModel Output: {model_response}")
            instruction = "Extract a simple sentence that I can use for a Google Search query from this string:\n"
            query = self.model.generate(instruction + model_response)

        # Remove unwanted newlines
        query = query.replace('\n', '')
        re.sub(r'[\n`´]', '', query)

        return query

    def _call_api(self, search_query: str) -> str:
        """Call the respective search API to get the search result."""
        match self.search_engine:
            case 'google':
                return self.serper_searcher.run(search_query)
            case 'wiki':
                return self.wiki_searcher.search(search_query)
            
    def sufficient_knowledge(
            self,
            claim: str, 
            past_searches: List[SearchResult]
        ) -> bool:
        """
        This function uses an LLM to evaluate the sufficiency of search_results.
        """
        knowledge = '\n'.join([s.result for s in past_searches if s.result is not None])
        knowledge = 'N/A' if not knowledge else knowledge

        instruction = ("Given the following information, determine if it is enough to conclusively decide "
                       "whether the claim is true or false with high certainty. If the information is sufficient, "
                       "respond 'sufficient'; otherwise, respond 'insufficient'. Respond with only one word.")
        input = f"{instruction}\n\nInformation:\n{knowledge}"
        model_decision = self.model.generate(input)
        if model_decision.lower() == "sufficient":
            print("Sufficient Knowledge:")
            print(knowledge)
            print("For Claim: ")
            print(claim)
            return True
        else:
            return False
