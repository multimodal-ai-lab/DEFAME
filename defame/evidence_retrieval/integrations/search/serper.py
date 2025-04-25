# Major modifications applied by Technical University of Darmstadt, FG Multimodal Grounded Learning.
# Copyright 2024 Google LLC


"""Class for querying the Google Serper API."""

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import requests

from config.globals import api_keys
from defame.common import logger
from defame.evidence_retrieval.integrations.search.common import SearchResults, Query, WebSource
from defame.utils.parsing import get_base_domain

_SERPER_URL = 'https://google.serper.dev'


@dataclass
class GoogleSearchResults(SearchResults):
    answer: str = None  # sometimes, Google provides a comprehensive answer
    knowledge_graph: str = None

    def __str__(self):
        if self.n_sources == 0:
            return "No search results found."
        else:
            text = "**Google Search Results**\n\n"
            if self.answer:
                text += f"Answer: {self.answer}\n\n"
            if self.knowledge_graph:
                text += f"Knowledge Graph: {self.knowledge_graph}\n\n"
            if self.sources:
                text += "Sources:\n" + "\n\n".join(map(str, self.sources))
            return text

    def __repr__(self):
        return (f"GoogleSearchResults(n_sources={len(self.sources)}, "
                f"has_answer={self.answer is not None}, "
                f"has_knowledge_graph={self.knowledge_graph is not None}, "
                f"sources={self.sources})")


class SerperAPI:
    """Wrapper for the Serper API, handling the communication."""
    def __init__(self,
                 gl: str = 'us',
                 hl: str = 'en',
                 tbs: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.serper_api_key = api_keys["serper_api_key"]
        self.gl = gl
        self.hl = hl
        self.tbs = tbs

    def search(self, query: Query) -> Optional[GoogleSearchResults]:
        """Run query through GoogleSearch and parse result."""
        assert self.serper_api_key, 'Missing serper_api_key.'
        assert query, 'Query must not be None.'
        assert query.text, 'Query text must not be None.'

        if query.end_date is not None:
            end_date = query.end_date.strftime('%d/%m/%Y')
            tbs = f"cdr:1,cd_min:1/1/1900,cd_max:{end_date}"
        else:
            tbs = self.tbs

        search_type = "image" if query.has_image() else "search"

        output = self._call_serper_api(
            query.text,
            gl=self.gl,
            hl=self.hl,
            tbs=tbs,
            search_type=search_type,
        )
        answer, knowledge_graph, web_sources = self._parse_results(output, query)
        return GoogleSearchResults(sources=web_sources, answer=answer,
                                   knowledge_graph=knowledge_graph, query=query)

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
                        logger.critical(error_msg)
                        raise RuntimeError(error_msg)

            except requests.exceptions.Timeout:
                sleep_time = min(sleep_time * 2, 600)
                sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
                logger.warning(f"Unable to reach Serper API: Connection timed out. "
                               f"Retrying after {sleep_time} seconds.")
                time.sleep(sleep_time)

        if response is None:
            raise ValueError('Failed to get a response from Serper API.')

        response.raise_for_status()
        search_results = response.json()
        return search_results

    def _parse_results(self, response: dict[Any, Any], query: Query) -> (str, str, list[WebSource]):
        """Parse results from API response."""
        answer = _parse_answer_box(response)
        knowledge_graph = _parse_knowledge_graph(response)
        sources = self._parse_sources(response, query)
        return answer, knowledge_graph, sources

    def _parse_sources(self, response: dict, query: Query) -> list[WebSource]:
        # TODO: Process sitelinks
        sources = []
        result_key = "images" if query.has_image() else "organic"
        filtered_results = filter_unique_results_by_domain(response[result_key])
        if result_key in response:
            for i, result in enumerate(filtered_results):
                if len(sources) >= query.limit:  # somehow the num param does not restrict requests.post image search results
                    break

                url = result.get("link") if result_key == "organic" else result.get("imageUrl")
                if not url:
                    continue

                title = result.get('title')

                try:
                    result_date = datetime.strptime(result['date'], "%b %d, %Y").date()
                except (ValueError, KeyError):
                    result_date = None
                sources.append(WebSource(reference=url, release_date=result_date, title=title))
        return sources


serper_api = SerperAPI()


def _parse_answer_box(response: dict) -> Optional[str]:
    """Parses the "answer box" which Google sometimes returns."""
    # TODO: If answer_box contains a source ('link' and 'snippet'), add it to the other sources
    if answer_box := response.get('answerBox'):
        answer = []
        if answer_raw := answer_box.get('answer'):
            answer.append(answer_raw)
        if snippet := answer_box.get('snippet'):
            answer.append(snippet)
        if link := answer_box.get('link'):
            answer.append(link)
        if snippet_highlighted := answer_box.get('snippetHighlighted'):
            answer.append(str(snippet_highlighted))
        return "\n".join(answer)


def _parse_knowledge_graph(response: dict) -> Optional[str]:
    # TODO: Test this
    if kg := response.get('knowledgeGraph'):
        knowledge_graph = []
        if title := kg.get('title'):
            knowledge_graph.append(title)
        if entity_type := kg.get('type'):
            knowledge_graph.append(f'Type: {entity_type}')
        if description := kg.get('description'):
            knowledge_graph.append(description)
        if attributes := kg.get('attributes'):
            for attribute, value in attributes.items():
                knowledge_graph.append(f'{attribute}: {value}')
        return "\n".join(knowledge_graph)


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


if __name__ == "__main__":
    example_query = Query(
        text="What is the first element in the periodic table?",
        limit=5,
        end_date=datetime(2025, 3, 5)
    )
    results = serper_api.search(example_query)
    print(results)

    # Test the cache (result should appear much faster)
    start = time.time()
    results = serper_api.search(example_query)
    end = time.time()
    print(f"Second search with same query took {end - start:.3f} seconds.")
