from abc import ABC

from infact.common import logger
from infact.common.misc import Query
from infact.utils.console import yellow
from .common import SearchResult


class SearchAPI(ABC):
    """Abstract base class for all local and remote search APIs."""
    name: str
    is_free: bool
    is_local: bool

    def __init__(self):
        self.total_searches = 0
        assert self.name is not None

    def _before_search(self, query: Query):
        self.total_searches += 1
        logger.log(yellow(f"Searching {self.name} with query: {query}"))

    def search(self, query: Query) -> SearchResult:
        """Runs the API by submitting the query and obtaining a list of search results."""
        self._before_search(query)
        return self._call_api(query)

    def _call_api(self, query: Query) -> SearchResult:
        raise NotImplementedError()
