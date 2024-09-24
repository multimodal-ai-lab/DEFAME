"""Class for querying the Google Serper API."""

import random
import time
from datetime import datetime
from typing import Any, Optional, Literal
import re

from PIL import Image as PillowImage
import requests
from io import BytesIO
from bs4 import BeautifulSoup

from config.globals import api_keys
from infact.common.misc import Query, WebSource
from infact.tools.search.remote_search_api import RemoteSearchAPI
from infact.common.medium import Image, media_registry

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

    def _call_api(self, query: Query) -> list[WebSource]:
        """Run query through GoogleSearch and parse result."""
        assert self.serper_api_key, 'Missing serper_api_key.'
        assert query, 'Searching Google with empty query'

        if query.end_date is not None:
            end_date = query.end_date.strftime('%d/%m/%Y')
            tbs = f"cdr:1,cd_min:1/1/1900,cd_max:{end_date}"
        else:
            tbs = self.tbs

        results = self._call_serper_api(
            query.text,
            gl=self.gl,
            hl=self.hl,
            num=query.limit,
            tbs=tbs,
            search_type=query.search_type,
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
        if result_key in response:
            for i, result in enumerate(response[result_key]):
                if i >= query.limit: #somehow the num param does not restrict requests.post image search results
                    break
                text = result.get("snippet", "")
                url = result.get("link", "")
                image_url = result.get("imageUrl", "")

                if result_key == "organic":
                    scraped_text = self.scrape_text_from_url(url)
                    if scraped_text:
                        keywords = re.findall(r'\b\w+\b', query.text.lower()) or query.text
                        relevant_content = filter_relevant_sentences(scraped_text, keywords)[:10]
                        relevant_text = ' '.join(relevant_content)
                        text = relevant_text or text
                    else:
                        continue

                elif result_key == "images":
                    try:
                        image_response = requests.get(image_url)
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

    def scrape_text_from_url(self, url):
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
            self.logger.info(f"HTTP error occurred while scraping {url}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            self.logger.info(f"Request exception occurred while scraping {url}: {req_err}")
        except Exception as e:
            self.logger.info(f"An unexpected error occurred while scraping {url}: {e}")
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