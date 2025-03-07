import os
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from PIL import Image as pillowImage
from google.cloud import vision

from config.globals import google_service_account_key_path
from defame.common import logger
from defame.common.medium import Image
from defame.common.misc import ImageQuery, WebSource
from defame.tools.search.common import ReverseSearchResult
from defame.tools.search.remote_search_api import RemoteSearchAPI
from defame.tools.search.remote_search_api import scrape, is_fact_checking_site, is_unsupported_site


class GoogleVisionAPI(RemoteSearchAPI):
    """Class for performing image reverse search using Google Cloud Vision API."""
    name = "google_vision"

    def __init__(self, activate_cache: bool = True, **kwargs):
        super().__init__(activate_cache=activate_cache, **kwargs)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_service_account_key_path.as_posix()
        self.client = vision.ImageAnnotatorClient()

    def _call_api(self, query: ImageQuery) -> ReverseSearchResult:
        """Run image reverse search through Google Vision API and parse results."""
        assert query.image, 'Image path or URL is required for image search.'

        image = vision.Image(content=query.image.get_base64_encoded())
        response = self.client.web_detection(image=image)
        if response.error.message:
            logger.warning(f"{response.error.message}\nCheck Google Cloud Vision API documentation for more info.")

        return self._parse_results(response.web_detection, query)

    def _parse_results(self, web_detection: vision.WebDetection, query: ImageQuery) -> ReverseSearchResult:
        """Parse Google Vision API web detection results into SearchResult instances."""

        # Web Entities
        web_entities = {}
        for entity in web_detection.web_entities:
            web_entities[entity.description] = entity.score

        # Best Guess Labels
        best_guess_labels = []
        if web_detection.best_guess_labels:
            for label in web_detection.best_guess_labels:
                if label.label:
                    best_guess_labels.append(label.label)

        # Pages Relevant Images
        web_sources = []
        filtered_pages = filter_unique_stem_pages(web_detection.pages_with_matching_images)
        for page in filtered_pages:
            url = page.url
            if is_fact_checking_site(url):
                logger.log(f"Skipping fact-checking website: {url}")
                continue

            if is_unsupported_site(url):
                logger.log(f"Skipping unsupported website: {url}")
                continue

            title = f'Found exact image on website with title: {page.page_title}' if hasattr(page,
                                                                                             'page_title') else "Found exact image on website"
            scraped = scrape(url)

            if scraped:
                web_sources.append(WebSource(
                    url=url,
                    title=title,
                    content=str(scraped),
                    query=query,
                    rank=len(web_sources) + 1
                ))

        return ReverseSearchResult(sources=web_sources, entities=web_entities, best_guess_labels=best_guess_labels)

    def _download_image(self, image_url: str) -> Optional[Image]:
        """Download an image from a URL and return it as a Pillow Image."""
        try:
            response = requests.get(image_url, timeout=7)
            response.raise_for_status()
            return Image(pillow_image=pillowImage.open(BytesIO(response.content)).convert('RGB'))
        except Exception as e:
            logger.log(f"Failed to download or open image from {image_url}: {e}")
            return None


def get_base_domain(url):
    """
    Extracts the base domain from a given URL, ignoring common subdomains like 'www' and 'm'.
    
    Args:
        url (str): The URL to extract the base domain from.
    
    Returns:
        str: The base domain (e.g., 'facebook.com').
    """
    netloc = urlparse(url).netloc

    # Remove common subdomains like 'www.' and 'm.'
    if netloc.startswith('www.') or netloc.startswith('m.'):
        netloc = netloc.split('.', 1)[1]  # Remove the first part (e.g., 'www.', 'm.')

    return netloc


def filter_unique_stem_pages(pages):
    """
    Filters pages to ensure only one page per website base domain is included 
    (e.g., 'facebook.com' regardless of subdomain), 
    and limits the total number of pages to the specified limit.
    
    Args:
        pages (list): List of pages with matching images.
        limit (int): Maximum number of pages to keep.
    
    Returns:
        list: Filtered list of pages.
    """
    unique_domains = set()
    filtered_pages = []

    for page in pages:
        base_domain = get_base_domain(page.url)

        # Check if we already have a page from this base domain
        if base_domain not in unique_domains:
            unique_domains.add(base_domain)
            filtered_pages.append(page)

    return filtered_pages


if __name__ == "__main__":
    query = ImageQuery(
        image=Image(pillow_image=pillowImage.open(
            "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/InFact/out/temp/media/516.jpg").convert('RGB')),
        search_type="search"
    )
    api = GoogleVisionAPI()
    result = api._call_api(query)
    print(result)
