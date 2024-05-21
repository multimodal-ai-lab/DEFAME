import dataclasses
from typing import Optional, Sequence, List

from common import utils
from common.modeling import Model
from common.shared_config import serper_api_key
from safe.config import num_searches, debug_safe, max_steps, max_retries
from safe.prompts.prompt import SearchPrompt
from safe.tools.query_serper import SerperAPI
from safe.tools.wiki_dump import WikiDumpAPI


@dataclasses.dataclass()
class SearchResult:
    query: str
    result: str


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
                if verbose:
                    print("next_search: ", next_search)
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
        """Get the next query from the model."""
        # Construct the prompt tasking the model to produce a search query
        knowledge = '\n'.join([s.result for s in past_searches if s.result is not None])
        knowledge = 'N/A' if not knowledge else knowledge
        past_queries = '\n'.join([s.query for s in past_searches])
        past_queries = 'N/A' if not past_queries else past_queries
        search_prompt = SearchPrompt(claim, knowledge, past_queries,
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

        return SearchResult(query=query, result=self._call_api(query))

    def post_process_query(self, model_response: str) -> str:
        """
        Processes the model response, ensures correct formatting, and adjusts the response if needed.
        """
        # Check if there is no query in the model response
        print("No query was found in output - likely due to wrong formatting.\nModel Output: {model_response}")
        # Adjust the model response
        adjustment_instruct = "Extract a simple sentence that I can use for a Google Search Query from this string:\n"
        adjusted_model_response = self.model.generate(adjustment_instruct + model_response)
        print(f'Extracted Query: {adjusted_model_response}')
        return adjusted_model_response

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
