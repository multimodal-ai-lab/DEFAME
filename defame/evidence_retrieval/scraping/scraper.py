"""A web scraping module to retrieve the contents of ANY website."""
import asyncio
from typing import Optional

import aiohttp
import requests
from ezmm import MultimodalSequence, Image

from config.globals import firecrawl_url
from defame.common import logger
from defame.evidence_retrieval.integrations import RETRIEVAL_INTEGRATIONS
from defame.evidence_retrieval.integrations.search import WebSource
from defame.evidence_retrieval.scraping.excluded import (is_unsupported_site, is_relevant_content,
                                                         is_fact_checking_site)
from defame.evidence_retrieval.scraping.util import scrape_naive, find_firecrawl, firecrawl_is_running, log_error_url, \
    resolve_media_hyperlinks
from defame.utils.parsing import get_domain
from defame.utils.requests import download, is_image_url

FIRECRAWL_URLS = [
    firecrawl_url,
    "http://localhost:3002",
    "http://firecrawl:3002",
    "http://0.0.0.0:3002",
]


class Scraper:
    """Takes any URL and tries to scrape its contents. If the URL belongs to a platform
    requiring an API and the API integration is implemented (e.g. X, Reddit etc.), the
    respective API will be used instead of direct HTTP requests."""

    firecrawl_url: Optional[str]

    def __init__(self, allow_fact_checking_sites: bool = True):
        self.allow_fact_checking_sites = allow_fact_checking_sites

        self.locate_firecrawl()
        if not self.firecrawl_url:
            logger.warning(f"❌ Unable to locate Firecrawl! It is not running at: {firecrawl_url}")

        self.n_scrapes = 0

    def locate_firecrawl(self):
        """Scans a list of URLs (included the user-specified one) to find a
        running Firecrawl instance."""
        self.firecrawl_url = find_firecrawl(FIRECRAWL_URLS)
        if self.firecrawl_url:
            logger.log(f"✅ Detected Firecrawl running at {self.firecrawl_url}.")

    def scrape_sources(self, sources: list[WebSource]) -> None:
        """Retrieves the contents for the given web sources and saves them
        into the respective web source object."""
        # Only keep sources that weren't scraped yet
        sources = [s for s in sources if not s.is_loaded()]

        if sources:
            urls = [s.url for s in sources]
            scrape_results = self.scrape_multiple(urls)
            for source, scraped in zip(sources, scrape_results):
                source.content = scraped

    def scrape_multiple(self, urls: list[str]) -> list[MultimodalSequence | None]:
        """Scrapes each URL concurrently. Synchronous wrapper for _scrape_multiple()."""
        return asyncio.run(self._scrape_multiple(urls))

    async def _scrape_multiple(self, urls: list[str]) -> list[MultimodalSequence | None]:
        tasks = [self._scrape(url) for url in urls]
        return await asyncio.gather(*tasks)

    def scrape(self, url: str) -> Optional[MultimodalSequence]:
        """Scrapes the contents of the specified webpage. Synchronous wrapper for _scrape()."""
        return asyncio.run(self._scrape(url))

    async def _scrape(self, url: str) -> Optional[MultimodalSequence]:
        # Check exclusions first
        if is_unsupported_site(url):
            logger.log(f"Skipping unsupported site: {url}")
            return None
        if not self.allow_fact_checking_sites and is_fact_checking_site(url):
            logger.log(f"Skipping fact-checking site: {url}")
            return None

        # Identify and use any applicable integration to retrieve the URL contents
        scraped = _retrieve_via_integration(url)
        if scraped:
            return scraped

        # Check if URL points to a media file. If yes, download accordingly TODO: extend to videos/audios
        if is_image_url(url):
            try:
                image = Image(binary_data=download(url))
                scraped = MultimodalSequence([image])
            except Exception:
                pass

        # Use Firecrawl to scrape from the URL
        if not scraped:
            # Try to find Firecrawl again if necessary
            if self.firecrawl_url is None:
                self.locate_firecrawl()

            if self.firecrawl_url:
                if firecrawl_is_running(self.firecrawl_url):
                    scraped = await self._scrape_firecrawl(url)
                else:
                    logger.error(f"Firecrawl stopped running! No response from {firecrawl_url}!. "
                                 f"Falling back to Beautiful Soup until Firecrawl is available again.")
                    self.firecrawl_url = None

        # If the scrape still was not successful, use naive Beautiful Soup scraper
        if not scraped:
            scraped = scrape_naive(url)

        if scraped:
            self.n_scrapes += 1
            return scraped

    async def _scrape_firecrawl(self, url: str) -> Optional[MultimodalSequence]:
        """Scrapes the given URL using Firecrawl. Returns a Markdown-formatted
        multimedia snippet, containing any (relevant) media from the page."""
        assert self.firecrawl_url is not None

        headers = {
            'Content-Type': 'application/json',
        }
        json_data = {
            "url": url,
            "formats": ["markdown"],
            "timeout": 15 * 60 * 1000,  # waiting time in milliseconds for Firecrawl to process the job
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.firecrawl_url + "/v1/scrape",
                                        json=json_data,
                                        headers=headers,
                                        timeout=10 * 60) as response:  # Firecrawl scrapes usually take 2 to 4s, but a 1700-page PDF takes 5 min

                    if response.status != 200:
                        logger.log(f"Failed to scrape {url}")
                        error_message = f"Failed to scrape {url} - Status code: {response.status} - Reason: {response.reason}"
                        log_error_url(url, error_message)
                        match response.status:
                            case 402:
                                logger.log(f"Error 402: Access denied.")
                            case 403:
                                logger.log(f"Error 403: Forbidden.")
                            case 408:
                                logger.warning(f"Error 408: Timeout! Firecrawl overloaded or Webpage did not respond.")
                            case 409:
                                logger.log(f"Error 409: Access denied.")
                            case 500:
                                logger.log(f"Error 500: Server error.")
                            case _:
                                logger.log(f"Error {response.status}: {response.reason}.")
                        logger.log("Skipping that URL.")
                        return None

                    json = await response.json()
                    success = json["success"]
                    if success and "data" in json:
                        data = json["data"]
                        text = data.get("markdown")
                        return resolve_media_hyperlinks(text)
                    else:
                        error_message = f"Unable to read {url}. No usable data in response."
                        logger.info(f"Unable to read {url}. Skipping it.")
                        logger.info(str(json))
                        log_error_url(url, error_message)
                        return None

        except (requests.exceptions.RetryError, requests.exceptions.ConnectionError):
            logger.error(f"Firecrawl is not running!")
            return None
        except requests.exceptions.Timeout:
            error_message = "Firecrawl failed to respond in time! This can be due to server overload."
            logger.warning(f"{error_message}\nSkipping the URL {url}.")
            log_error_url(url, error_message)
            return None
        except Exception as e:
            error_message = f"Exception: {repr(e)}"
            logger.info(repr(e))
            logger.info(f"Unable to scrape {url} with Firecrawl. Skipping...")
            log_error_url(url, error_message)
            return None


def _retrieve_via_integration(url: str) -> Optional[MultimodalSequence]:
    domain = get_domain(url)
    if domain in RETRIEVAL_INTEGRATIONS:
        integration = RETRIEVAL_INTEGRATIONS[domain]
        return integration.retrieve(url)


scraper = Scraper()

if __name__ == "__main__":
    print(scrape_naive("https://www.independent.co.uk/news/world/africa/sahara-desert-snow-first-40-years-rare-photos-atlas-mountains-algeria-karim-bouchetata-a7488056.html"))
    print(scraper.scrape("https://www.independent.co.uk/news/world/africa/sahara-desert-snow-first-40-years-rare-photos-atlas-mountains-algeria-karim-bouchetata-a7488056.html"))
    print(scraper.scrape_multiple([
        "https://www.washingtonpost.com/video/national/cruz-calls-trump-clinton-two-new-york-liberals/2016/04/07/da3b78a8-fcdf-11e5-813a-90ab563f0dde_video.html",
        "https://cdn.pixabay.com/photo/2017/11/08/22/28/camera-2931883_1280.jpg",
        "https://www.tagesschau.de/ausland/asien/libanon-israel-blauhelme-nahost-102.html",
        "https://www.zeit.de/politik/ausland/2024-10/wolodymyr-selenskyj-berlin-olaf-scholz-militaerhilfe",
        "https://pixum-cms.imgix.net/7wL8j3wldZEONCSZB9Up6B/d033b7b6280687ce2e4dfe2d4147ff93/fab_mix_kv_perspektive_foto_liegend_desktop__3_.png?auto=compress,format&trim=false&w=2000",
    ]))
