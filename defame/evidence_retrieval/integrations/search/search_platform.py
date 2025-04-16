from abc import ABC
from typing import Optional

from defame.common import logger
from defame.utils.console import yellow
from .common import SearchResults, Query


class SearchPlatform(ABC):
    """Abstract base class for all local and remote search platforms."""
    name: str
    is_local: bool
    description: str

    def __init__(self):
        self.n_searches = 0
        assert self.name is not None

    def _before_search(self, query: Query):
        self.n_searches += 1
        logger.log(yellow(f"Searching {self.name} with query: {query}"))

    def search(self, query: Query | str) -> Optional[SearchResults]:
        """Runs the API by submitting the query and obtaining a list of search results."""
        if isinstance(query, str):
            query = Query(text=query)
        self._before_search(query)
        return self._call_api(query)

    def _call_api(self, query: Query) -> Optional[SearchResults]:
        raise NotImplementedError()

    def reset(self):
        """Resets the search API to its initial state (if applicable) and sets all stats to zero."""
        self.n_searches = 0

    @property
    def stats(self) -> dict:
        return {"Searches (API Calls)": self.n_searches}
