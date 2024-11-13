"""Class for querying the Google Serper API."""

import random
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Optional, Literal

import requests
from PIL import Image as PillowImage

from config.globals import api_keys
from infact.common.medium import Image
from infact.common.misc import Query, WebSource
from infact.tools.search.remote_search_api import RemoteSearchAPI, scrape, is_fact_checking_site, is_unsupported_site
from .common import SearchResult
from .google_vision_api import get_base_domain

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
        self.result_key_for_type = {
            'news': 'news',
            'places': 'places',
            'images': 'images',
            'search': 'organic',
        }

    def _call_api(self, query: Query) -> SearchResult:
        """Run query through GoogleSearch and parse result."""
        assert self.serper_api_key, 'Missing serper_api_key.'
        assert query, 'Searching Google with empty query'

        if query.end_date is not None:
            end_date = query.end_date.strftime('%d/%m/%Y')
            tbs = f"cdr:1,cd_min:1/1/1900,cd_max:{end_date}"
        else:
            tbs = self.tbs

        output = self._call_serper_api(
            query.text,
            gl=self.gl,
            hl=self.hl,
            tbs=tbs,
            search_type=query.search_type,
        )
        web_sources = self._parse_results(output, query)
        return SearchResult(web_sources)

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
        response, num_tries, sleep_time = None, 0, 0

        while not response and num_tries < max_retries:
            num_tries += 1
            try:
                response = requests.post(
                    f'{_SERPER_URL}/{search_type}', headers=headers, params=params, timeout=3,
                )

                if response.status_code == 400:
                    message = response.json().get('message')
                    if message == "Not enough credits":
                        error_msg = "No Serper API credits left anymore! Please recharge the Serper account."
                        self.logger.critical(error_msg)
                        raise RuntimeError(error_msg)

            except requests.exceptions.Timeout:
                sleep_time = min(sleep_time * 2, 600)
                sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
                self.logger.warning(f"Unable to reach Serper API: Connection timed out. "
                                    f"Retrying after {sleep_time} seconds.")
                time.sleep(sleep_time)

            # except Exception as e:
            #     self.logger.warning("Failed to call Serper API:\n" + repr(e))

        if response is None:
            raise ValueError('Failed to get a response from Serper API.')

        response.raise_for_status()
        search_results = response.json()
        return search_results

    def _parse_results(self, response: dict[Any, Any], query: Query) -> list[WebSource]:
        """Parse results from API response."""

        snippets = []
        if response.get('answerBox'):
            answer_box = response.get('answerBox', {})
            answer = answer_box.get('answer')
            snippet = answer_box.get('snippet')
            snippet_highlighted = answer_box.get('snippetHighlighted')

            if answer and isinstance(answer, str):
                snippets.append(answer)
            if snippet and isinstance(snippet, str):
                snippets.append(snippet.replace('\n', ' '))
            if snippet_highlighted:
                snippets.append(snippet_highlighted)

        if response.get('knowledgeGraph'):
            kg = response.get('knowledgeGraph', {})
            title = kg.get('title')
            entity_type = kg.get('type')
            description = kg.get('description')

            if entity_type:
                snippets.append(f'{title}: {entity_type}.')

            if description:
                snippets.append(description)

            for attribute, value in kg.get('attributes', {}).items():
                snippets.append(f'{title} {attribute}: {value}.')

        results = []
        result_key = self.result_key_for_type[query.search_type]
        filtered_results = filter_unique_results_by_domain(response[result_key])
        if result_key in response:
            for i, result in enumerate(filtered_results):
                limit = query.limit or self.max_search_results
                if len(results) >= limit:  # somehow the num param does not restrict requests.post image search results
                    break
                text = result.get("snippet", "")
                url = result.get("link", "")
                if is_fact_checking_site(url):
                    self.logger.log(f"Skipping fact-checking website: {url}")
                    continue
                if is_unsupported_site(url):
                    self.logger.log(f"Skipping unsupported website: {url}")
                    continue

                image_url = result.get("imageUrl", "")
                title = result.get('title')

                if result_key == "organic":
                    text = scrape(url=url, logger=self.logger)
                    if not text:
                        continue
                    else:
                        text = str(text)

                elif result_key == "images":
                    try:
                        image_response = requests.get(image_url, timeout=10)
                        image = Image(pillow_image=PillowImage.open(BytesIO(image_response.content)))
                        if image:
                            text += f"\n{image.reference}"

                    except Exception as e:
                        self.logger.log(f"Failed to download or open image: {e}")

                try:
                    result_date = datetime.strptime(result['date'], "%b %d, %Y").date()
                except (ValueError, KeyError):
                    result_date = None
                results.append(WebSource(url=url, text=text, query=query, rank=i, date=result_date))

        return results


def filter_unique_results_by_domain(results):
    """
    Filters the results to ensure only one result per website base domain is included
    (e.g., 'facebook.com' regardless of subdomain).

    Args:
        results (list): List of result dictionaries from the search result.

    Returns:
        list: Filtered list of unique results by domain.
    """
    unique_domains = set()
    filtered_results = []

    for result in results:
        url = result.get("link", "")  # Extract URL from the result dictionary
        if not url:
            continue  # Skip if no URL is found

        base_domain = get_base_domain(url)

        # Add the result if we haven't seen this domain before
        if base_domain not in unique_domains:
            unique_domains.add(base_domain)
            filtered_results.append(result)

    return filtered_results
