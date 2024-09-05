"""Class for querying the Google Serper API."""

import random
import time
from typing import Any, Optional, Literal

import requests

from infact.common.results import SearchResult
from infact.tools.search.remote_search_api import RemoteSearchAPI
from config.globals import api_keys

_SERPER_URL = 'https://google.serper.dev'
NO_RESULT_MSG = 'No good Google Search result was found'


class SerperAPI(RemoteSearchAPI):
    """Class for querying the Google Serper API."""
    name = "google"

    def __init__(self,
                 gl: str = 'us',
                 hl: str = 'en',
                 tbs: Optional[str] = None,
                 search_type: Literal['news', 'search', 'places', 'images'] = 'search',
                 **kwargs):
        super().__init__(**kwargs)
        self.serper_api_key = api_keys["serper_api_key"]
        self.gl = gl
        self.hl = hl
        self.tbs = tbs
        self.search_type = search_type
        self.total_searches = 0
        self.result_key_for_type = {
            'news': 'news',
            'places': 'places',
            'images': 'images',
            'search': 'organic',
        }

    def _call_api(self, query: str, limit: int, **kwargs: Any) -> list[SearchResult]:
        """Run query through GoogleSearch and parse result."""
        assert self.serper_api_key, 'Missing serper_api_key.'
        assert query, 'Searching Google with empty query'
        results = self._call_serper_api(
            query,
            gl=self.gl,
            hl=self.hl,
            num=limit,
            tbs=self.tbs,
            search_type=self.search_type,
            **kwargs,
        )
        return self._parse_results(results, query)

    def _call_serper_api(
            self,
            search_term: str,
            search_type: str = 'search',
            max_retries: int = 20,
            **kwargs: Any,
    ) -> dict[Any, Any]:
        """Run query through Google Serper."""
        headers = {
            'X-API-KEY': self.serper_api_key or '',
            'Content-Type': 'application/json',
        }
        params = {
            'q': search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }
        response, num_fails, sleep_time = None, 0, 0

        while not response and num_fails < max_retries:
            try:
                self.total_searches += 1
                response = requests.post(
                    f'{_SERPER_URL}/{search_type}', headers=headers, params=params
                )
            except AssertionError as e:
                raise e
            except Exception:  # pylint: disable=broad-exception-caught
                response = None
                num_fails += 1
                sleep_time = min(sleep_time * 2, 600)
                sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
                time.sleep(sleep_time)

        if not response:
            raise ValueError('Failed to get result from Google Serper API')

        response.raise_for_status()
        search_results = response.json()
        return search_results

    def _parse_results(self, response: dict[Any, Any], query: str) -> list[SearchResult]:
        """Parse results from API response."""
        # if response.get('answerBox'):
        #     answer_box = response.get('answerBox', {})
        #     answer = answer_box.get('answer')
        #     snippet = answer_box.get('snippet')
        #     snippet_highlighted = answer_box.get('snippetHighlighted')
        #
        #     if answer and isinstance(answer, str):
        #         snippets.append(answer)
        #     if snippet and isinstance(snippet, str):
        #         snippets.append(snippet.replace('\n', ' '))
        #     if snippet_highlighted:
        #         snippets.append(snippet_highlighted)
        #
        # if response.get('knowledgeGraph'):
        #     kg = response.get('knowledgeGraph', {})
        #     title = kg.get('title')
        #     entity_type = kg.get('type')
        #     description = kg.get('description')
        #
        #     if entity_type:
        #         snippets.append(f'{title}: {entity_type}.')
        #
        #     if description:
        #         snippets.append(description)
        #
        #     for attribute, value in kg.get('attributes', {}).items():
        #         snippets.append(f'{title} {attribute}: {value}.')

        results = []
        result_key = self.result_key_for_type[self.search_type]
        if result_key in response:
            for i, result in enumerate(response[result_key]):
                if "snippet" not in result:
                    text = "NONE"
                elif "title" not in result:
                    text = f"{result['snippet']}"
                else:
                    text = f"{result['title']}: {result['snippet']}"
                url = result["link"]
                results.append(SearchResult(url=url, text=text, query=query, rank=i))

        return results
