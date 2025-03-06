import io
import re
from typing import Optional

import requests
from PIL import Image as PillowImage, UnidentifiedImageError
from bs4 import BeautifulSoup

from config.globals import temp_dir
from defame.common import MultimediaSnippet, logger, Image
from defame.utils.parsing import md, get_markdown_hyperlinks, is_image_url


def read_urls_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()


def scrape_naive(url: str) -> Optional[MultimediaSnippet]:
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


def resolve_media_hyperlinks(text: str) -> Optional[MultimediaSnippet]:
    """Identifies up to MAX_MEDIA_PER_PAGE image URLs, downloads the images and replaces the
    respective Markdown hyperlinks with their proper image reference."""
    if text is None:
        return None
    hyperlinks = get_markdown_hyperlinks(text)
    media_count = 0
    for hypertext, url in hyperlinks:
        if is_image_url(url):
            try:
                # TODO: Handle very large images like: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144225/campfire_oli_2018312_lrg.jpg
                # Download the image
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    img = PillowImage.open(io.BytesIO(response.content))
                    image = Image(pillow_image=img)  # TODO: Check for duplicates
                    # Replace the Markdown hyperlink
                    text = re.sub(rf"!?\[{hypertext}]\({url}\)", f"{hypertext} {image.reference}", text)
                    media_count += 1
                    if media_count >= MAX_MEDIA_PER_PAGE:
                        break
                    else:
                        continue

            except (
            requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout,
            requests.exceptions.TooManyRedirects):
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


MAX_MEDIA_PER_PAGE = 32  # Any media URLs in a webpage exceeding this limit will be ignored.
