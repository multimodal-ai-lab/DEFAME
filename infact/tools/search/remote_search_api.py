import os
import pickle
import re
from pathlib import Path
from typing import Optional
import io

import requests
from bs4 import BeautifulSoup
from PIL import Image as PillowImage, UnidentifiedImageError
from notebook.auth import passwd

from config.globals import result_base_dir
from infact.common.misc import Query
from infact.common import Logger, MultimediaSnippet, Image
from infact.tools.search.search_api import SearchAPI
from infact.tools.search.common import SearchResult
from infact.utils.parsing import md, get_markdown_hyperlinks, is_image_url


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
        self.cache: dict[Query, SearchResult] = {}
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

    def _add_to_cache(self, query: Query, search_result: SearchResult):
        """Adds the given query-results pair to the cache."""
        self.cache[query] = search_result
        self._save_data()

    def search(self, query: Query) -> SearchResult:
        # Try to load from cache
        if self.search_cached_first:
            cache_results = self.search_cache(query)
            if cache_results:
                self.cache_hit += 1
                return cache_results

        # Run actual search
        search_result = super().search(query)
        self._add_to_cache(query, search_result)
        return search_result

    def search_cache(self, query: Query) -> SearchResult:
        """Search the local in-memory data for matching results."""
        if query in self.cache:
            return self.cache[query]


def scrape(url, logger) -> MultimediaSnippet:
    """Scrapes the contents of the specified webpage."""
    # TODO: Handle social media links (esp. Twitter/X, YouTube etc.) differently
    scraped = scrape_firecrawl(url, logger)
    if scraped:
        return scraped
    else:
        return MultimediaSnippet(scrape_naive(url, logger))


def scrape_naive(url, logger):
    """Fallback scraping script."""
    # TODO: Also scrape images
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    try:
        page = requests.get(url, headers=headers, timeout=5)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')
        # text = soup.get_text(separator='\n', strip=True)
        if soup.article:
            # News articles often use the <article> tag to mark article contents
            soup = soup.article
        # Turn soup object into a Markdown-formatted string
        text = md(soup)
        text = postprocess_scraped(text)
        return text
    except requests.exceptions.Timeout:
        logger.info(f"Timeout occurred while scraping {url}")
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


def postprocess_scraped(text: str) -> str:
    # Remove any excess whitespaces
    text = re.sub(r' {2,}', ' ', text)

    # remove any excess newlines
    text = re.sub(r'(\n *){3,}', '\n\n', text)

    return text


def scrape_firecrawl(url: str, logger: Logger) -> Optional[MultimediaSnippet]:
    """Scrapes the given URL using Firecrawl. Returns a Markdown-formatted
    multimedia snippet, containing any (relevant) media from the page."""
    headers = {
        'Content-Type': 'application/json',
    }
    firecrawl_url = "http://localhost:3002/v1/scrape"
    json_data = {
        "url": url,
        "formats": ["markdown", "html"]
    }

    try:
        response = requests.post(firecrawl_url,
                                 json=json_data,
                                 headers=headers,
                                 timeout=10)
    except (requests.exceptions.RetryError, ConnectionRefusedError, requests.exceptions.ConnectionError):
        logger.error(f"Firecrawl is not running! Falling back...")
        return None
    except Exception as e:
        logger.info(repr(e))
        logger.info(f"Unable to read {url}. Skipping...")
        return None

    if response.status_code in [402, 403, 409]:
        error_message = response.json().get('error', 'Unknown error occurred')
        logger.log(f'Failed to scrape URL {url}.\nError {response.status_code}: {response.reason}. Message: {error_message}.')
        return None
    elif response.status_code == 500:
        error_message = response.json().get('error', 'Unknown error occurred')
        logger.info(f'Failed to scrape URL {url}.\nError {response.status_code}: {response.reason}. Message: {error_message}.')
        return None


    success = response.json()["success"]
    if success:
        data = response.json()["data"]
        text = data["markdown"]
        return _resolve_media_hyperlinks(text)
    else:
        logger.info(str(response))
        logger.info(f"Unable to read {url}. Skipping...")
        return None


def _resolve_media_hyperlinks(text: str) -> MultimediaSnippet:
    """Identifies all image URLs, downloads the images and replaces the
    respective Markdown hyperlinks with their proper image reference."""
    hyperlinks = get_markdown_hyperlinks(text)
    for hypertext, url in hyperlinks:
        # Check if URL is an image URL
        if is_image_url(url):
            try:
                # Download the image
                response = requests.get(url, stream=True, timeout=5)
                if response.status_code == 200:
                    img = PillowImage.open(io.BytesIO(response.content))
                    image = Image(pillow_image=img)  # TODO: Check for duplicates
                    # Replace the Markdown hyperlink
                    text = text.replace(f"[{hypertext}]({url})", f"{hypertext} {image.reference}")
                    continue

            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
                # Webserver is not reachable (anymore)
                pass

            except UnidentifiedImageError as e:
                print(f"Unable to download image from {url}.")
                print(e)
                # Image has an incompatible format. Skip it.
                pass

            finally:
                # Remove the hyperlink, just keep the hypertext
                text = text.replace(f"[{hypertext}]({url})", "")

        # TODO: Resolve videos and audios
    return MultimediaSnippet(text)


if __name__ == '__main__':
    print("Running scrapes...")
    urls_to_scrape = [
        "https://nypost.com/2024/10/11/us-news/meteorologists-hit-with-death-threats-after-debunking-hurricane-conspiracy-theories/",
        "https://www.tagesschau.de/ausland/asien/libanon-israel-blauhelme-nahost-102.html",
        "https://www.zeit.de/politik/ausland/2024-10/wolodymyr-selenskyj-berlin-olaf-scholz-militaerhilfe",
        "https://edition.cnn.com/2024/10/07/business/property-damange-hurricane-helene-47-billion/index.html"
    ]
    for url in urls_to_scrape:
        scraped = scrape_firecrawl(url, None)
        if scraped:
            print(scraped, "\n\n\n")
        else:
            print("Scrape failed.")
