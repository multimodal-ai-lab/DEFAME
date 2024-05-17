import dataclasses
from typing import Optional, Sequence

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

    def __init__(self, search_tool: str, model: Model):
        assert search_tool in ["serper", "wiki"]
        self.search_tool = search_tool
        self.model = model

        self.serper_searcher = SerperAPI(serper_api_key, k=num_searches)
        #self.wiki_searcher = WikiDumpAPI()

        self.max_steps = max_steps
        self.max_retries = max_retries
        self.debug = debug_safe

    def search(self, claim, verbose: bool = False) -> Sequence[SearchResult]:
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

        return search_results

    def _maybe_get_next_search(self,
                              claim: str,
                              past_searches: list[SearchResult],
                              verbose: Optional[bool] = False,
                              ) -> SearchResult | None:
        """Get the next query from the model."""
        knowledge = '\n'.join([s.result for s in past_searches])
        knowledge = 'N/A' if not knowledge else knowledge
        search_prompt = SearchPrompt(claim, knowledge, open_source=self.model.open_source)
        model_response = self.model.generate(str(search_prompt), do_debug=self.debug).replace('"', '')
        if model_response.startswith("I cannot"):
            if verbose: 
                utils.print_guard()
            model_response = claim
        query = utils.extract_first_code_block(model_response, ignore_language=True)
        if not query:
            query = utils.post_process_query(model_response, model=self.model)

        return SearchResult(query=query, result=self._call_api(query))


    def _call_api(self, search_query: str) -> str:
        """Call the respective search API to get the search result."""
        match self.search_tool:
            case 'serper':
                return self.serper_searcher.run(search_query)
            case 'wiki':
                return self.wiki_searcher.search(search_query)
