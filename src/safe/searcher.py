import dataclasses
from typing import Optional, Sequence, List, Any
import re
from logging import Logger

from common import utils
from common.modeling import Model
from common.shared_config import serper_api_key
from safe.config import num_searches, debug_safe, max_steps, max_retries
from safe.prompts.prompt import SearchPrompt, SummarizePrompt
from safe.tools.query_serper import SerperAPI
from safe.tools.duckduckgo import DuckDuckGo
from safe.tools.wiki_dump import WikiDumpAPI
from eval.logging import print_log


@dataclasses.dataclass()
class SearchResult:
    query: str
    result: str

    def __str__(self):
        return f"SearchResult(\n\tquery='{self.query}'\n\tresult='{self.result}'\n)"


class Searcher:
    """Searches the specified resource (Google, Wikipedia, ...) for evidence."""

    def __init__(self, search_engine: str, model: Model):
        assert search_engine in ["google", "wiki", "duckduck"]
        self.search_engine = search_engine
        self.model = model

        self.serper_searcher = SerperAPI(serper_api_key, k=num_searches)
        self.wiki_searcher = WikiDumpAPI()
        self.duckduck_searcher = DuckDuckGo(max_results=num_searches)

        self.max_steps = max_steps
        self.max_retries = max_retries
        self.debug = debug_safe

# TODO: rank the websites according to their credibility like MUSE
    def search(
            self, 
            claim, 
            limit_search=True,
            summarize=True,
            verbose: bool = False,
            logger: Optional[Logger] = None,
    ) -> Sequence[SearchResult]:
        
        search_results = []
        for _ in range(self.max_steps):
            next_search, num_tries = None, 0

            while not next_search and num_tries <= self.max_retries:
                next_search = self._maybe_get_next_search(claim, search_results, summarize=summarize, verbose=verbose, logger=logger)
                num_tries += 1

            if next_search is None or not next_search.result:
                if verbose:
                    utils.maybe_print_error(f'Unsuccessful parsing for `next_search` try #{num_tries}. Likely wrong search. Try again...')
                if logger:
                    print_log(logger, f'Unsuccessful parsing for `next_search` try #{num_tries}. Likely wrong search. Try again...')
                    print_log(logger, f'next_search: {next_search}')
            else:
                search_results.append(next_search)

            if limit_search and self.sufficient_knowledge(claim, search_results, verbose=verbose, logger=logger):
                break
        return search_results

    def _maybe_get_next_search(self,
                               claim: str,
                               past_searches: List[SearchResult],
                               summarize: bool = True,
                               verbose: bool = False,
                               logger: Optional[Logger] = None,
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
            if logger:
                print_log(logger, f"Model hit the guardrails with prompt:\n {search_prompt}")
            model_response = '```' + claim + '```'
        query = utils.extract_first_code_block(model_response, ignore_language=True)
        if not query:
            query = self.post_process_query(model_response, verbose=verbose, logger=logger)

        # Avoid casting the same, previously used query again
        if query in past_queries:
            return
    
        result = self._call_api(query, verbose=verbose, logger=logger)

        if logger:
            print_log(logger, f'Query: {query}')
            print_log(logger, f'Result: {result}')

        # Avoid duplicate results
        #TODO the following two lines seem superfluous as the likelihood 
        # of getting (exact) duplicate results is low for these long strings?
        if result in past_results:
            result = None  # But keep query to avoid future duplicates

        # If result is too long, summarize it (to avoid hitting the context length limit)
        if summarize and result is not None and len(result) > 728:
            summarize_prompt = SummarizePrompt(query, result)
            result = self.model.generate(str(summarize_prompt), do_debug=self.debug)
            if verbose:
                print("Summarized result:", result)
            if logger:
                print_log(logger, f"Summarized result: {result}")
        
        search_result = SearchResult(query=query, result=result)
        if verbose:
            print("Found", search_result)
        if logger:
            print_log(logger, f'Found: {search_result}')

        return search_result

    def post_process_query(
            self, 
            model_response: str, 
            verbose: bool = False,
            logger: Optional[Logger] = None,
    ) -> str:
        """
        Processes the model response to extract the query. Ensures correct formatting
        and adjusts the response if needed.
        """

        # If query extraction was unsuccessful, use the LLM to extract the query from the response
        if verbose:
            print(f"No query was found in output - likely due to wrong formatting.\nModel Output: {model_response}")
        if logger:
            print_log(logger, f"No query was found in output - likely due to wrong formatting. Model Output: {model_response}")
        
        instruction = "Extract a simple sentence that I can use for a Google Search query from this string:\n"
        query = self.model.generate(instruction + model_response)

        # Remove unwanted newlines
        query = query.replace('\n', '')
        re.sub(r'[\n`´]', '', query)

        return query

    def _call_api(self, search_query: str,verbose: bool = False, logger: Optional[Logger] = None) -> str:
        """Call the respective search API to get the search result."""
        match self.search_engine:
            case 'google':
                return self.serper_searcher.run(search_query)
            case 'wiki':
                return self.wiki_searcher.search(search_query)
            case 'duckduck':
                return self.duckduck_searcher.run(search_query, verbose=verbose, logger=logger)
            
    def sufficient_knowledge(
            self,
            claim: str, 
            past_searches: List[SearchResult],
            verbose: bool = False,
            logger: Optional[Logger] = None,
        ) -> bool:
        """
        This function uses an LLM to evaluate the sufficiency of search_results.
        """ 
        knowledge = '\n'.join([s.result for s in past_searches if s.result is not None])
        knowledge = 'N/A' if not knowledge else knowledge

        instruction = ("Given the following INFORMATION, determine if it is enough to conclusively decide "
                       "whether the CLAIM is true or false with high certainty. If the INFORMATION is sufficient, "
                       "respond 'sufficient'. Otherwise, respond 'insufficient'. "
                       "If you are in doubt or need more information, respond 'insufficient'. "
                       "Respond with only one word.")
        input = f"{instruction}\INFORMATION:\n{knowledge}\CLAIM:{claim}"
        model_decision = self.model.generate(input)
        if model_decision.lower() == "sufficient":
            if verbose:
                print(f"Sufficient knowledge:\n{knowledge}\nFor claim:\n{claim}")
            if logger:
                print_log(logger, f"Sufficient knowledge: {knowledge}")
                print_log(logger, f"For claim: {claim}")
            return True
        else:
            return False
