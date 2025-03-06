"""A web scraping module to retrieve the contents of ANY website."""
from typing import Optional

import requests

from config.globals import firecrawl_url
from defame.common import MultimediaSnippet, logger
from defame.evidence_retrieval.integrations.scraping.excluded import (is_unsupported_site, is_relevant_content,
                                                                      is_fact_checking_site)
from .util import scrape_naive, find_firecrawl, firecrawl_is_running, log_error_url, resolve_media_hyperlinks

firecrawl_urls = [
    firecrawl_url,
    "http://localhost:3002",
    "http://firecrawl:3002",
    "http://0.0.0.0:3002",
]


class Scraper:
    """Takes any URL and tries to scrape its contents. If the URL belongs to a platform
    requiring an API and the API integration is implemented (e.g. X, Reddit etc.), the
    respective API will be used instead of direct HTTP requests."""

    def __init__(self, allow_fact_checking_sites: bool = True):
        self.allow_fact_checking_sites = allow_fact_checking_sites

        # Verify that Firecrawl is running
        self.warning_issued = False
        self.firecrawl_url = find_firecrawl(firecrawl_urls)
        if self.firecrawl_url:
            logger.info(f"✅ Detected Firecrawl running at {self.firecrawl_url}.")
        else:
            logger.error(f"❌ Unable to locate Firecrawl! It is not running at {firecrawl_url}.")
            self.warning_issued = True

    def scrape(self, url: str) -> Optional[MultimediaSnippet]:
        """Scrapes the contents of the specified webpage."""
        # Check exclusions first
        if is_unsupported_site(url):
            logger.log(f"Skipping unsupported site: {url}")
            return None
        if not self.allow_fact_checking_sites and is_fact_checking_site(url):
            logger.log(f"Skipping fact-checking site: {url}")
            return None

        # Identify any applicable integration to use existing APIs for retrieval
        # TODO: Insert integration calls here

        if self.firecrawl_url and firecrawl_is_running(self.firecrawl_url):
            scraped = self._scrape_firecrawl(url)
        else:
            if not self.warning_issued:
                logger.error(f"Firecrawl is not running at {firecrawl_url}! Falling back to Beautiful Soup.")
                self.warning_issued = True
            scraped = scrape_naive(url)

        if scraped and is_relevant_content(str(scraped)):
            return scraped
        else:
            return None

    def _scrape_firecrawl(self, url: str) -> Optional[MultimediaSnippet]:
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
            response = requests.post(self.firecrawl_url + "/v1/scrape",
                                     json=json_data,
                                     headers=headers,
                                     timeout=10 * 60)  # Firecrawl scrapes usually take 2 to 4s, but a 1700-page PDF takes 5 min
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

        if response.status_code != 200:
            logger.log(f"Failed to scrape {url}")
            error_message = f"Failed to scrape {url} - Status code: {response.status_code} - Reason: {response.reason}"
            log_error_url(url, error_message)
            match response.status_code:
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
                    logger.log(f"Error {response.status_code}: {response.reason}.")
            logger.log("Skipping that URL.")
            return None

        success = response.json()["success"]
        if success and "data" in response.json():
            data = response.json()["data"]
            text = data.get("markdown")
            return resolve_media_hyperlinks(text)
        else:
            error_message = f"Unable to read {url}. No usable data in response."
            logger.info(f"Unable to read {url}. Skipping it.")
            logger.info(str(response.content))
            log_error_url(url, error_message)
            return None


scraper = Scraper()
