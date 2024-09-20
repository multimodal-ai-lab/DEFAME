from abc import ABC

from infact.common import Logger
from infact.common.misc import Query, WebSource
from infact.utils.console import yellow


class SearchAPI(ABC):
    """Abstract base class for all local and remote search APIs."""
    name: str
    is_free: bool
    is_local: bool

    def __init__(self, logger: Logger = None):
        self.logger = logger
        self.total_searches = 0

    def _before_search(self, query: Query):
        self.total_searches += 1
        if self.logger is not None:
            self.logger.log(yellow(f"Searching {self.name} with query: {query}"))

    def search(self, query: Query) -> list[WebSource]:
        """Runs the API by submitting the query and obtaining a list of search results."""
        self._before_search(query)
        return self._call_api(query)

    def _call_api(self, query: Query) -> list[WebSource]:
        raise NotImplementedError()
