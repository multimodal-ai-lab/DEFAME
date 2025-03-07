import os
import pickle
import sqlite3
from pathlib import Path
from typing import Optional

from config.globals import temp_dir
from defame.evidence_retrieval.integrations.search_engines.common import SearchResults, Query
from defame.evidence_retrieval.integrations.search_engines.search_api import SearchAPI


class RemoteSearchAPI(SearchAPI):
    """Any search engine that leverages an external/non-local API. Employs a caching
    mechanism to improve efficiency."""
    is_local = False

    def __init__(self,
                 activate_cache: bool = True,
                 max_search_results: int = 10,
                 **kwargs):
        super().__init__()
        self.max_search_results = max_search_results

        self.search_cached_first = activate_cache
        self.cache_file_name = f"{self.name}_cache.db"
        self.path_to_cache = Path(temp_dir) / self.cache_file_name
        self.cache_hit = 0

        if self.search_cached_first:
            if is_new := not self.path_to_cache.exists():
                os.makedirs(os.path.dirname(self.path_to_cache), exist_ok=True)
            self.conn = sqlite3.connect(self.path_to_cache, timeout=10, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.cur = self.conn.cursor()
            if is_new:
                self._init_db()

    def _init_db(self):
        """Initializes a clean, new DB."""
        stmt = f"""
            CREATE TABLE Query(hash TEXT PRIMARY KEY, results BLOB);
        """
        self.cur.execute(stmt)
        self.conn.commit()

    def _add_to_cache(self, query: Query, search_result: SearchResults):
        """Adds the given query-results pair to the cache."""
        stmt = f"""
            INSERT INTO Query(hash, results)
            VALUES (?, ?);
        """
        self.cur.execute(stmt, (hash(query), pickle.dumps(search_result)))
        self.conn.commit()

    def _get_from_cache(self, query: Query) -> Optional[SearchResults]:
        """Search the local in-memory data for matching results."""
        stmt = f"""
            SELECT results FROM Query WHERE hash = ?;
        """
        response = self.cur.execute(stmt, (hash(query),))
        result = response.fetchone()
        if result is not None:
            return pickle.loads(result[0])

    def search(self, query: Query) -> Optional[SearchResults]:
        # Try to load from cache
        if self.search_cached_first:
            cache_results = self._get_from_cache(query)
            if cache_results:
                self.cache_hit += 1
                return cache_results

        # Run actual search
        search_result = super().search(query)
        self._add_to_cache(query, search_result)
        return search_result
