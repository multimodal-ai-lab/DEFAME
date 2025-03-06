import os
from typing import Sequence

from google.cloud import vision

from config.globals import google_service_account_key_path
from defame.common import logger
from defame.common.medium import Image
from defame.common.misc import ImageQuery, WebSource
from defame.evidence_retrieval.integrations.search_engines.common import ReverseSearchResult
from defame.evidence_retrieval.integrations.search_engines.remote_search_api import RemoteSearchAPI
from defame.utils.parsing import get_base_domain


def _parse_results(web_detection: vision.WebDetection, query: ImageQuery) -> ReverseSearchResult:
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
        title = f'Found exact image on website with title: {page.page_title}' if \
            hasattr(page, 'page_title') else "Found exact image on website"

        # TODO: Move scraping to tool
        from defame.evidence_retrieval.scraping.scraper import scraper
        scraped = scraper.scrape(url)

        if scraped:
            web_sources.append(WebSource(
                url=url,
                title=title,
                data=str(scraped),
                query=query,
                rank=len(web_sources) + 1
            ))

    return ReverseSearchResult(sources=web_sources, entities=web_entities, best_guess_labels=best_guess_labels)


class GoogleVisionAPI(RemoteSearchAPI):
    """Class for performing reverse image search (RIS) using Google Cloud Vision API."""
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

        return _parse_results(response.web_detection, query)


def filter_unique_stem_pages(pages: Sequence):
    """
    Filters pages to ensure only one page per website base domain is included 
    (e.g., 'facebook.com' regardless of subdomain), 
    and limits the total number of pages to the specified limit.
    
    Args:
        pages (list): List of pages with matching images.
    
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
    example_query = ImageQuery(
        image=Image("in/example/sahara.webp"),
        search_type="search"
    )
    api = GoogleVisionAPI()
    result = api.search(example_query)
    print(result)
