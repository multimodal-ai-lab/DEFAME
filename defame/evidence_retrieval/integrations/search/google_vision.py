import os
from dataclasses import dataclass
from typing import Sequence

from ezmm import Image
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import vision

from config.globals import google_service_account_key_path
from defame.common import logger
from defame.evidence_retrieval.integrations.search.common import WebSource, Query, SearchMode, SearchResults
from defame.utils.parsing import get_base_domain


@dataclass
class GoogleRisResults(SearchResults):
    """Reverse Image Search (RIS) results. Ship with additional object detection
    information next to the list of sources."""
    entities: dict[str, float]  # mapping between entity description and confidence score
    best_guess_labels: list[str]

    @property
    def exact_matches(self):
        return self.sources

    def __str__(self):
        text = "**Reverse Image Search Results**"

        if self.entities:
            text += f"\n\nIdentified entities (confidence in parenthesis):\n"
            text += "\n".join(f"- {name} ({confidence * 100:.0f} %)"
                              for name, confidence in self.entities.items())

        if self.best_guess_labels:
            text += f"\n\nBest guess about the topic of the image: {', '.join(self.best_guess_labels)}."

        if self.exact_matches:
            text += "\n\nThe same image was found in the following sources:\n"
            text += "\n".join(map(str, self.exact_matches))

        return text

    def __repr__(self):
        return (f"RisResults(n_exact_matches={len(self.exact_matches)}, "
                f"n_entities={len(self.entities)}, "
                f"n_best_guess_labels={len(self.best_guess_labels)})")


class GoogleVisionAPI:
    """Wraps the Google Cloud Vision API for performing reverse image search (RIS)."""

    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_service_account_key_path.as_posix()
        try:
            self.client = vision.ImageAnnotatorClient()
        except DefaultCredentialsError:
            logger.warning(f"❌ No or invalid Google Cloud API credentials at "
                           f"{google_service_account_key_path.as_posix()}.")
        else:
            logger.log(f"✅ Successfully connected to Google Cloud Vision API.")

    def search(self, query: Query) -> GoogleRisResults:
        """Run image reverse search through Google Vision API and parse results."""
        assert query.has_image(), "Google Vision API requires an image in the query."

        image = vision.Image(content=query.image.get_base64_encoded())
        response = self.client.web_detection(image=image)
        if response.error.message:
            logger.warning(f"{response.error.message}\nCheck Google Cloud Vision API documentation for more info.")

        return _parse_results(response.web_detection, query)


google_vision_api = GoogleVisionAPI()


def _parse_results(web_detection: vision.WebDetection, query: Query) -> GoogleRisResults:
    """Parse Google Vision API web detection results into SearchResult instances."""

    # Web Entities
    web_entities = {}
    for entity in web_detection.web_entities:
        if entity.description:
            web_entities[entity.description] = entity.score

    # Best Guess Labels
    best_guess_labels = []
    if web_detection.best_guess_labels:
        for label in web_detection.best_guess_labels:
            if label.label:
                best_guess_labels.append(label.label)

    # Pages with relevant images
    web_sources = []
    filtered_pages = _filter_unique_stem_pages(web_detection.pages_with_matching_images)
    for page in filtered_pages:
        url = page.url
        title = page.__dict__.get("page_title")
        web_source = WebSource(reference=url, title=title)
        web_sources.append(web_source)

    return GoogleRisResults(sources=web_sources,
                            query=query,
                            entities=web_entities,
                            best_guess_labels=best_guess_labels)


def _filter_unique_stem_pages(pages: Sequence):
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
