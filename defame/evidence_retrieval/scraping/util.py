import re
from typing import Optional

import requests
from PIL import UnidentifiedImageError
from bs4 import BeautifulSoup
from ezmm import MultimodalSequence

from config.globals import temp_dir
from defame.common import logger
from defame.utils.parsing import md, get_markdown_hyperlinks
from defame.utils.requests import download_image, is_image_url

MAX_MEDIA_PER_PAGE = 32  # Any media URLs in a webpage exceeding this limit will be ignored.


def read_urls_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()


def scrape_naive(url: str) -> Optional[MultimodalSequence]:
    """Fallback scraping script."""
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
        return MultimodalSequence(text)
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


def resolve_media_hyperlinks(text: str) -> Optional[MultimodalSequence]:
    """Identifies up to MAX_MEDIA_PER_PAGE image URLs, downloads the images and replaces the
    respective Markdown hyperlinks with their proper image reference."""
    # TODO: Resolve videos and audios

    if text is None:
        return None
    hyperlinks = get_markdown_hyperlinks(text)
    media_count = 0
    for hypertext, url in hyperlinks:
        if is_image_url(url):
            try:
                image = download_image(url)
                if image:
                    # Replace the Markdown hyperlink with the image reference
                    text = re.sub(rf"!?\[{re.escape(hypertext)}]\({re.escape(url)}\)",
                                  f"{hypertext} {image.reference}", text)
                    media_count += 1
                    if media_count >= MAX_MEDIA_PER_PAGE:
                        break
                    else:
                        continue

            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.TooManyRedirects):
                # Webserver is not reachable (anymore)
                pass

            except UnidentifiedImageError as e:
                logger.warning(f"Unable to download image from {url}.")
                logger.warning(e)
                # Image has an incompatible format. Skip it.

            except Exception as e:
                logger.warning(f"Unable to download image from {url}.")
                logger.warning(e)

            finally:
                # Remove the hyperlink, just keep the hypertext
                text = text.replace(f"[{hypertext}]({url})", "")

    return MultimodalSequence(text)


def log_error_url(url: str, message: str):
    error_log_file = temp_dir.parent / "crawl_error_log.txt"
    with open(error_log_file, "a") as f:
        f.write(f"{url}: {message}\n")


def find_firecrawl(urls):
    for url in urls:
        if firecrawl_is_running(url):
            return url
    return None


def firecrawl_is_running(url):
    """Returns True iff Firecrawl is running."""
    try:
        response = requests.get(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.RetryError):
        return False
    return response.status_code == 200


if __name__ == "__main__":
    hyperlink = "![Snowfall in the Sahara desert](https://modernsciences.org/wp-content/uploads/2022/12/Snowfall-in-the-Sahara-desert_-an-unusual-weather-phenomenon-80x42.png)"
    print(resolve_media_hyperlinks(hyperlink))
