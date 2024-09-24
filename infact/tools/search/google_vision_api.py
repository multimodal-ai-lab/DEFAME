# google_vision_api.py

import os
import requests
from io import BytesIO
from infact.common.medium import Image
from PIL import Image as pillowImage
from google.cloud import vision
from typing import Any, List
import re

from infact.tools.search.remote_search_api import RemoteSearchAPI, scrape_text_from_url
from infact.common.misc import ImageQuery, WebSource
from config.globals import api_keys

class GoogleVisionAPI(RemoteSearchAPI):
    """Class for performing image reverse search using Google Cloud Vision API."""
    name = "google_vision"

    def __init__(self, logger: Any = None, activate_cache: bool = True, **kwargs):
        super().__init__(logger=logger, activate_cache=activate_cache, **kwargs)
        self.google_vision_key = api_keys["google_vision_key"]
        if not self.google_vision_key:
            raise ValueError("GOOGLE_VISION_KEY is not set in the API keys configuration.")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/google_service_account_key.json"
        self.client = vision.ImageAnnotatorClient()
        self.total_searches = 0

    def _call_api(self, query: ImageQuery) -> list[WebSource]:
        """Run image reverse search through Google Vision API and parse results."""
        assert query.image, 'Image path or URL is required for image search.'

        results = self._call_vision_api(query)
        return self._parse_results(results, query)

    def _call_vision_api(self, query: ImageQuery) -> vision.WebDetection:
        """Call the Google Vision API with the provided image."""
        image = vision.Image(content=query.image.get_base64_encoded())
        response = self.client.web_detection(image=image)
        self.total_searches += 1
        if response.error.message:
            raise Exception(f"{response.error.message}\nCheck Google Cloud Vision API documentation for more info.")
        return response.web_detection

    def _parse_results(self, web_detection: vision.WebDetection, query: ImageQuery) -> list[WebSource]:
        """Parse Google Vision API web detection results into SearchResult instances."""
        results = []

        # Best Guess Labels
        if web_detection.best_guess_labels:
            for label in web_detection.best_guess_labels[:query.limit]:
                text = f"Best guess label: {label.label}"
                results.append(WebSource(
                    url="",  # No specific URL for labels
                    text=text,
                    query=query,
                    rank=len(results) + 1
                ))

        # Pages with Matching Images
        for page in web_detection.pages_with_matching_images[:query.limit]:
            url = page.url
            title = page.page_title if hasattr(page, 'page_title') else ""
            text = f"Page Title: {title}"
            results.append(WebSource(
                url=url,
                text=text,
                query=query,
                rank=len(results) + 1
            ))

            # Full Matching Images
            for image in page.full_matching_images[:query.limit]:
                image_url = image.url

                scraped_text = scrape_text_from_url(url=image_url, logger=self.logger)
                if scraped_text:
                    text = f"Full Matching Image at {image_url}:\n{scraped_text}"
                    results.append(WebSource(
                        url=image_url,
                        text=text,
                        query=query,
                        rank=len(results) + 1,
                    ))
                else:
                    continue

            # Partial Matching Images
            for image in page.partial_matching_images[:query.limit]:
                image_url = image.url
                image = self._download_image(image_url)
                scraped_text = scrape_text_from_url(url=image_url, logger=self.logger)
                if image:
                    text = f"Partial Matching Image {image.reference} at {image_url}:\n{scraped_text}"
                    results.append(WebSource(
                        url=image_url,
                        text=text,
                        query=query,
                        rank=len(results) + 1,
                    ))
                elif scraped_text:
                    text = f"Partial Matching Image at {image_url}:\n{scraped_text}"
                    results.append(WebSource(
                        url=image_url,
                        text=text,
                        query=query,
                        rank=len(results) + 1,
                    ))
                else:
                    continue

        # Visually Similar Images
        for image in web_detection.visually_similar_images[:query.limit]:
            image_url = image.url
            image = self._download_image(image_url)
            scraped_text = scrape_text_from_url(url=image_url, logger=self.logger)
            if image:
                text = f"Visually similar Image {image.reference} at {image_url}:\n{scraped_text}"
                results.append(WebSource(
                    url=image_url,
                    text=text,
                    query=query,
                    rank=len(results) + 1,
                ))
            elif scraped_text:
                text = f"Visually similar Image at {image_url}:\n{scraped_text}"
                results.append(WebSource(
                    url=image_url,
                    text=text,
                    query=query,
                    rank=len(results) + 1,
                ))
            else:
                continue

        # Web Entities
        web_entities = ""
        for entity in web_detection.web_entities:
            description = entity.description
            score = entity.score
            if description:
                text = f"Web Entity: {description} (Score: {score})"
                web_entities = "\n".join([web_entities, text])
        results.append(WebSource(
            url="",  # No specific URL for entities
            text=web_entities,
            query=query,
            rank=len(results) + 1
        ))

        return results

    def _download_image(self, image_url: str) -> Image:
        """Download an image from a URL and return it as a Pillow Image."""
        try:
            response = requests.get(image_url, timeout=7)
            response.raise_for_status()
            return Image(pillowImage.open(BytesIO(response.content)).convert('RGB'))
        except Exception as e:
            self.logger.log(f"Failed to download or open image from {image_url}: {e}")
            return None
