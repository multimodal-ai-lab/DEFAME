import os
import pickle
import re
from pathlib import Path
from typing import Optional
import io
import time

import requests
from bs4 import BeautifulSoup
from PIL import Image as PillowImage, UnidentifiedImageError

from config.globals import temp_dir, firecrawl_url
from defame.common.misc import Query
from defame.common import logger, MultimediaSnippet, Image
from defame.tools.search.search_api import SearchAPI
from defame.tools.search.common import SearchResult
from defame.utils.parsing import md, get_markdown_hyperlinks, is_image_url, get_domain

MAX_MEDIA_PER_PAGE = 32  # Any media URLs in a webpage exceeding this limit will be ignored.

fact_checking_urls = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "truthorfiction.com",
    "fullfact.org",
    "leadstories.com",
    "hoax-slayer.net",
    "checkyourfact.com",
    "reuters.com/fact-check",
    "reuters.com/article/fact-check",
    "apnews.com/APFactCheck",
    "factcheck.afp.com",
    "poynter.org",
    "factcheck.ge",
    "vishvasnews.com",
    "boomlive.in",
    "altnews.in",
    "thequint.com/news/webqoof",
    "factcheck.kz",
    "data.gesis.org/claimskg/claim_review",
]

# These sites don't allow bot access/scraping. Must use a
# proprietary API or a different way to access them.
unsupported_domains = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "reddit.com",
    "ebay.com",
    "microsoft.com",
    "researchhub.com",
    "pinterest.com",
    "irs.gov"
]

block_keywords = [
        "captcha",
        "verify you are human",
        "access denied",
        "premium content",
        "403 Forbidden",
        "You have been blocked",
        "Please enable JavaScript",
        "I'm not a robot",
        "Are you a robot?",
        "Are you a human?",
    ]

unscrapable_urls = [
    "https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html",
    "https://data.news-leader.com/gas-price/",
    "https://www.wlbt.com/2023/03/13/3-years-later-mississippis-/",
    "https://edition.cnn.com/2021/01/11/business/no-fl",
    "https://www.thelugarcenter.org/ourwork-Bipart",
    "https://www.linkedin.com/pulse/senator-kelly-/",
    "http://libres.uncg.edu/ir/list-etd.aspx?styp=ty&bs=master%27s%20thesis&amt=100000",
    "https://www.washingtonpost.com/investigations/coronavirus-testing-denials/2020/03/",

]

class RemoteSearchAPI(SearchAPI):
    # TODO: Rewrite this for parallel access (maybe SQLite?)
    is_local = False

    def __init__(self,
                 activate_cache: bool = True,
                 max_search_results: int = 10,
                 **kwargs):
        super().__init__()
        self.max_search_results = max_search_results
        self.search_cached_first = activate_cache
        self.cache_file_name = f"{self.name}_cache.pckl"
        self.path_to_cache = os.path.join(Path(temp_dir) / self.cache_file_name)
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
        n_tries = 0
        while True:
            try:
                with open(self.path_to_cache, 'rb') as f:
                    self.cache = pickle.load(f)
                    return
            except Exception:
                time.sleep(1)
                n_tries += 1
                if n_tries >= 10:
                    raise

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
                #debug_list = [source.summary.text for source in cache_results.sources]
                #if any(debug_list):
                #    print(debug_list)
                for websource in cache_results.sources:
                    websource.summary = None # to ensure no summaries are taken from the cache
                return cache_results

        # Run actual search
        search_result = super().search(query)
        self._add_to_cache(query, search_result)
        return search_result

    def search_cache(self, query: Query) -> SearchResult:
        """Search the local in-memory data for matching results."""
        if query in self.cache:
            return self.cache[query]


def scrape(url: str) -> Optional[MultimediaSnippet]:
    """Scrapes the contents of the specified webpage."""
    if is_unsupported_site(url):
        logger.log(f"Skipping unsupported site {url}.")
        return None

    if _firecrawl_is_running():
        scraped = scrape_firecrawl(url)
        
    else:
        logger.warning(f"Firecrawl is not running! Falling back to Beautiful Soup.")
        scraped = scrape_naive(url)

    if scraped and is_relevant_content(str(scraped)):
        return scraped
    else:
        return None

def scrape_naive(url: str) ->  Optional[MultimediaSnippet]:
    """Fallback scraping script."""
    # TODO: Also scrape images
    headers = {
        'User-Agent': 'Mozilla/4.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    try:
        page = requests.get(url, headers=headers, timeout=5)

        # Handle any request errors
        if page.status_code == 403:
            logger.info(f"Forbidden URL: {url}")
            return None
        elif page.status_code == 404:
            return None
        page.raise_for_status()

        soup = BeautifulSoup(page.content, 'html.parser')
        # text = soup.get_text(separator='\n', strip=True)
        if soup.article:
            # News articles often use the <article> tag to mark article contents
            soup = soup.article
        # Turn soup object into a Markdown-formatted string
        text = md(soup)
        text = postprocess_scraped(text)
        return MultimediaSnippet(text)
    except requests.exceptions.Timeout:
        logger.info(f"Timeout occurred while naively scraping {url}")
    except requests.exceptions.HTTPError as http_err:
        logger.info(f"HTTP error occurred while doing naive scrape: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logger.info(f"Request exception occurred while scraping {url}: {req_err}")
    except Exception as e:
        logger.info(f"An unexpected error occurred while scraping {url}: {e}")
    return None


def postprocess_scraped(text: str) -> str:
    # Remove any excess whitespaces
    text = re.sub(r' {2,}', ' ', text)

    # remove any excess newlines
    text = re.sub(r'(\n *){3,}', '\n\n', text)

    return text


def scrape_firecrawl(url: str) -> Optional[MultimediaSnippet]:
    """Scrapes the given URL using Firecrawl. Returns a Markdown-formatted
    multimedia snippet, containing any (relevant) media from the page."""
    headers = {
        'Content-Type': 'application/json',
    }
    json_data = {
        "url": url,
        "formats": ["markdown"],
        "timeout": 15 * 1000,  # waiting time in milliseconds for Firecrawl to process the job
    }

    try:
        response = requests.post(firecrawl_url + "/v1/scrape",
                                 json=json_data,
                                 headers=headers,
                                 timeout=60)  # Firecrawl scrapes usually take 2 to 4s, but a 1700-page PDF takes 5 min
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
            case 402: logger.log(f"Error 402: Access denied.")
            case 403: logger.log(f"Error 403: Forbidden.")
            case 408: logger.warning(f"Error 408: Timeout! Firecrawl overloaded or Webpage did not respond.")
            case 409: logger.log(f"Error 409: Access denied.")
            case 500: logger.log(f"Error 500: Server error.")
            case _: logger.log(f"Error {response.status_code}: {response.reason}.")
        logger.log("Skipping that URL.")
        return None

    success = response.json()["success"]
    if success and "data" in response.json():
        data = response.json()["data"]
        text = data.get("markdown")
        return _resolve_media_hyperlinks(text)
    else:
        error_message = f"Unable to read {url}. No usable data in response."
        logger.info(f"Unable to read {url}. Skipping it.")
        logger.info(str(response.content))
        log_error_url(url, error_message)
        return None


def _resolve_media_hyperlinks(text: str) -> Optional[MultimediaSnippet]:
    """Identifies up to MAX_MEDIA_PER_PAGE image URLs, downloads the images and replaces the
    respective Markdown hyperlinks with their proper image reference."""
    if text is None:
        return None
    hyperlinks = get_markdown_hyperlinks(text)
    media_count = 0
    for hypertext, url in hyperlinks:
        # Check if URL is an image URL
        if is_image_url(url) and not is_fact_checking_site(url) and not is_unsupported_site(url):
            try:
                # TODO: Handle very large images like: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144225/campfire_oli_2018312_lrg.jpg
                # Download the image
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    img = PillowImage.open(io.BytesIO(response.content))
                    image = Image(pillow_image=img)  # TODO: Check for duplicates
                    # Replace the Markdown hyperlink
                    text = text.replace(f"[{hypertext}]({url})", f"{hypertext} {image.reference}")
                    media_count += 1
                    if media_count >= MAX_MEDIA_PER_PAGE:
                        break
                    else:
                        continue

            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.TooManyRedirects):
                # Webserver is not reachable (anymore)
                pass

            except UnidentifiedImageError as e:
                print(f"Unable to download image from {url}.")
                print(e)
                # Image has an incompatible format. Skip it.
                pass

            except Exception as e:
                print(f"Unable to download image from {url}.")
                print(e)
                pass

            finally:
                # Remove the hyperlink, just keep the hypertext
                text = text.replace(f"[{hypertext}]({url})", "")

        # TODO: Resolve videos and audios
    return MultimediaSnippet(text)


def _firecrawl_is_running():
    """Returns True iff Firecrawl is running."""
    try:
        response = requests.get(firecrawl_url)
    except (requests.exceptions.ConnectionError, requests.exceptions.RetryError):
        return False
    return response.status_code == 200


def is_fact_checking_site(url: str) -> bool:
    """Check if the URL belongs to a known fact-checking website."""
    # Check if the domain matches any known fact-checking website
    for site in fact_checking_urls:
        if site in url:
            return True
    return False


def is_unsupported_site(url: str) -> bool:
    """Checks if the URL belongs to a known unsupported website."""
    if (".gov" in url) or (url in unscrapable_urls):
        return True
    domain = get_domain(url)
    return domain in unsupported_domains


def is_relevant_content(content: str) -> bool:
    """Checks if the web scraping result contains relevant content or is blocked by a bot-catcher/paywall."""

    if not content:
        return False

    # Check for suspiciously short content (less than 500 characters might indicate blocking)
    if len(content.strip()) < 500:
        return False

    for keyword in block_keywords:
        if re.search(keyword, content, re.IGNORECASE):
            return False

    return True


def log_error_url(url: str, message: str):
        error_log_file = temp_dir.parent / "crawl_error_log.txt"
        with open(error_log_file, "a") as f:
            f.write(f"{url}: {message}\n")



if __name__ == '__main__':
    logger.info("Running scrapes with Firecrawl...")
    urls_to_scrape = [
        "https://www.washingtonpost.com/video/national/cruz-calls-trump-clinton-two-new-york-liberals/2016/04/07/da3b78a8-fcdf-11e5-813a-90ab563f0dde_video.html",
        "https://nypost.com/2024/10/11/us-news/meteorologists-hit-with-death-threats-after-debunking-hurricane-conspiracy-theories/",
        "https://www.tagesschau.de/ausland/asien/libanon-israel-blauhelme-nahost-102.html",
        "https://www.zeit.de/politik/ausland/2024-10/wolodymyr-selenskyj-berlin-olaf-scholz-militaerhilfe",
        "https://edition.cnn.com/2024/10/07/business/property-damange-hurricane-helene-47-billion/index.html"
    ]
    for url in urls_to_scrape:
        scraped = scrape_firecrawl(url)
        if scraped:
            print(scraped, "\n\n\n")
        else:
            logger.error("Scrape failed.")
