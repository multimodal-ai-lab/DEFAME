import json
import os
from datetime import datetime
from typing import List
from dataclasses import asdict

from safe.tools.search.search_api import SearchAPI, SearchResult
from common.shared_config import path_to_data
from eval.logger import EvaluationLogger

class RemoteSearchAPI(SearchAPI):
    is_local = False

    def __init__(self, logger: EvaluationLogger = None, path_to_data: str = path_to_data, search_local: bool = True, **kwargs):
        super().__init__(logger=logger)
        self.path_to_data = path_to_data
        self.search_local_first = search_local
        self.path_to_cache = os.path.join(self.path_to_data, "search_cache.json")
        self.local_searches = 0
        self.data = []
        self._initialize_datafile()
    
    def _initialize_datafile(self):
        if not os.path.exists(self.path_to_cache):
            os.makedirs(os.path.dirname(self.path_to_cache), exist_ok=True)
            with open(self.path_to_cache, 'w') as f:
                json.dump([], f)
        else:
            self._load_data()

    def _load_data(self):
        with open(self.path_to_cache, 'r') as f:
            self.data = json.load(f)
        
    def _save_results(self, results: List[SearchResult]):
        self.data.extend([asdict(result) for result in results])
        with open(self.path_to_cache, 'w') as f:
            json.dump(self.data, f, indent=4)

    def search(self, query: str, limit: int) -> List[SearchResult]:
        if self.search_local_first:
            self.local_searches += 1
            local_results = self.search_local(query)
            if local_results:
                return local_results

        results = super().search(query, limit)
        self._save_results(results)
        return results

    def search_local(self, query: str) -> List[SearchResult]:
        """Search the local in-memory data for matching results."""
        return [SearchResult(**result) for result in self.data if query.lower() in result['query'].lower()]
