from PIL import Image as PillowImage
from io import BytesIO
from typing import Optional, Any

import requests
from ezmm import Image

from defame.utils.parsing import is_image


# Mimic a browser to increase chances being accepted by servers
HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/123.0.0.0 Safari/537.36",
        }


def download_image(image_url: str) -> Optional[Image]:
    """Download an image from a URL and return it as an Image object."""
    # TODO: Handle very large images like: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144225/campfire_oli_2018312_lrg.jpg
    import pillow_avif  # Keep this import, as this adds AVIF file format support to pillow
    response = requests.get(image_url, stream=True, timeout=10, headers=HEADERS)
    response.raise_for_status()
    img = PillowImage.open(BytesIO(response.content))
    return Image(pillow_image=img)  # TODO: Check for duplicates


def download(url: str, max_size: int = None) -> Any:
    """Downloads a binary file from a given URL.
    TODO: Implement max file size."""
    response = requests.get(url, stream=True, headers=HEADERS)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
    return response.content


def is_image_url(url: str) -> bool:
    """Returns True iff the URL points at an accessible _pixel_ image file."""
    try:
        response = requests.head(url, timeout=2, allow_redirects=True, headers=HEADERS)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if content_type.startswith("image/"):
            return (not "svg" in response.headers.get('content-type') and
                    not "svg" in content_type and
                    not "eps" in content_type)
        elif content_type == "binary/octet-stream":
            # The content is a binary download stream. We need to download it
            # to determine the file type.
            binary_data = download(url)
            return is_image(binary_data)
        else:
            return False

    except Exception:
        return False


def is_media_url(url: str) -> bool:
    """TODO: Also check for videos and audios."""
    return is_image_url(url)
