import os
import pickle
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import re

from config.globals import result_base_dir
from infact.common.logger import Logger
from infact.common.misc import Query, WebSource
from infact.tools.search.search_api import SearchAPI


class RemoteSearchAPI(SearchAPI):
    is_local = False

    def __init__(self, logger: Logger = None,
                 activate_cache: bool = True,
                 **kwargs):
        super().__init__(logger=logger)
        self.search_cached_first = activate_cache
        self.cache_file_name = f"{self.name}_cache.pckl"
        self.path_to_cache = os.path.join(Path(result_base_dir) / self.cache_file_name)
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
        if query.search_type == 'image':
            from infact.tools.search.query_serper import SerperAPI
            assert isinstance(self, SerperAPI), "Need SerperAPI for image search"
        results = super().search(query)
        self._add_to_cache(query, results)
        return results

    def search_cache(self, query: Query) -> list[WebSource]:
        """Search the local in-memory data for matching results."""
        if query in self.cache:
            return self.cache[query]

def scrape_text_from_url(url, logger):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    try:
        page = requests.get(url, headers=headers)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except requests.exceptions.HTTPError as http_err:
        logger.info(f"HTTP error occurred while scraping {url}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logger.info(f"Request exception occurred while scraping {url}: {req_err}")
    except Exception as e:
        logger.info(f"An unexpected error occurred while scraping {url}: {e}")
    return ""

def filter_relevant_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant_sentences = []
    for sentence in sentences:
        score = sum(1 for word in keywords if word in sentence.lower())
        if score > 0:
            relevant_sentences.append((sentence, score))
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in relevant_sentences]