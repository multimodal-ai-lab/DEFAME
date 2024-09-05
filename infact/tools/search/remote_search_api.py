import json
import os
from dataclasses import asdict

from infact.common import SearchResult
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
        self.cache: list[dict] = []
        self._initialize_cache()

    def _initialize_cache(self):
        if not os.path.exists(self.path_to_cache):
            os.makedirs(os.path.dirname(self.path_to_cache), exist_ok=True)
            with open(self.path_to_cache, 'w') as f:
                json.dump([], f)
        else:
            self._load_data()

    def _load_data(self):
        with open(self.path_to_cache, 'r') as f:
            self.cache = json.load(f)

    def _add_to_cache(self, results: list[SearchResult]):
        """Adds the given search results to the cache."""
        for result in results:
            if result.query:
                self.cache.extend([asdict(result)])
            else:
                print(results)
                raise KeyError

        # Save cache file
        with open(self.path_to_cache, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if self.search_cached_first:
            cache_results = self.search_cache(query)
            if cache_results:
                self.cache_hit += 1
                return cache_results

        results = super().search(query, limit)
        self._add_to_cache(results)
        return results

    def search_cache(self, query: str) -> list[SearchResult]:
        """Search the local in-memory data for matching results."""
        return [SearchResult(**result) for result in self.cache if query.lower() in result['query'].lower()]
