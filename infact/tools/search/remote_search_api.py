import os
import pickle

from infact.common.misc import Query, WebSource
from infact.common.logger import Logger
from infact.tools.search.search_api import SearchAPI


class RemoteSearchAPI(SearchAPI):
    is_local = False
    file_name = "search_cache.json"

    def __init__(self, logger: Logger = None,
                 activate_cache: bool = True,
                 **kwargs):
        super().__init__(logger=logger)
        self.search_cached_first = activate_cache
        self.path_to_cache = os.path.join(logger.target_dir / self.file_name)
        self.cache_hit = 0
        self.cache: dict[Query, list[WebSource]] = {}
        self._initialize_cache()

    def _initialize_cache(self):
        if not os.path.exists(self.path_to_cache):
            os.makedirs(os.path.dirname(self.path_to_cache), exist_ok=True)
            self._save_data()
        else:
            self._load_data()

    def _load_data(self):
        with open(self.path_to_cache, 'rb') as f:
            self.cache = pickle.load(f)

    def _save_data(self):
        with open(self.path_to_cache, 'wb') as f:
            pickle.dump(self.cache, f)

    def _add_to_cache(self, query: Query, results: list[WebSource]):
        """Adds the given query-results pair to the cache."""
        self.cache[query] = results
        self._save_data()

    def search(self, query: Query) -> list[WebSource]:
        # Try to load from cache
        if self.search_cached_first:
            cache_results = self.search_cache(query)
            if cache_results:
                self.cache_hit += 1
                return cache_results

        results = super().search(query)
        self._add_to_cache(query, results)
        return results

    def search_cache(self, query: Query) -> list[WebSource]:
        """Search the local in-memory data for matching results."""
        if query in self.cache:
            return self.cache[query]
