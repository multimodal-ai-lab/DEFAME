import os
from dataclasses import dataclass
from typing import Sequence

from google.cloud import vision

from config.globals import google_service_account_key_path
from defame.common import logger
from defame.common.medium import Image
from defame.evidence_retrieval.integrations.search_engines.common import WebSource, Query, SearchMode, SearchResults
from defame.evidence_retrieval.integrations.search_engines.remote_search_api import RemoteSearchAPI
from defame.utils.parsing import get_base_domain


@dataclass
class RisResults(SearchResults):
    """Reverse Image Search (RIS) results. Ship with additional object detection
    information next to the list of sources."""
    entities: dict[str, float]  # mapping between entity description and confidence score
    best_guess_labels: list[str]

    def __str__(self):
        text = "Google Vision's outputs"
        if self.entities:
            text += f"\nIdentified web entities (confidence in parenthesis):\n"
            text += "\n".join(f"{description} ({confidence * 100:.0f} %)"
                                     for description, confidence in self.entities.items())
        if self.best_guess_labels:
            text += (f"\nBest guess about the topic of "
                            f"the image is {', '.join(self.best_guess_labels)}.\n Exact image matches found at:")
        return "**Reverse Search Result** The exact image was found in the following sources:\n\n" + "\n\n".join(
            map(str, self.sources))


def _parse_results(web_detection: vision.WebDetection, query: Query) -> RisResults:
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
                reference=url,
                title=title,
                content=scraped
            ))

    return RisResults(sources=web_sources, query=query, entities=web_entities, best_guess_labels=best_guess_labels)


class GoogleVisionAPI(RemoteSearchAPI):
    """Class for performing reverse image search (RIS) using Google Cloud Vision API."""
    name = "google_vision"

    def __init__(self, activate_cache: bool = True, **kwargs):
        super().__init__(activate_cache=activate_cache, **kwargs)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_service_account_key_path.as_posix()
        self.client = vision.ImageAnnotatorClient()

    def _call_api(self, query: Query) -> RisResults:
        """Run image reverse search through Google Vision API and parse results."""
        assert query.has_image(), "Google Vision API requires an image in the query."

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
    example_query = Query(
        image=Image("in/example/sahara.webp"),
        search_mode=SearchMode.REVERSE,
    )
    api = GoogleVisionAPI()
    result = api.search(example_query)
    print(result)
