from PIL import Image as PillowImage
from io import BytesIO
from typing import Optional, Any

import requests

from defame.common import Image


def download_image(image_url: str) -> Optional[Image]:
    """Download an image from a URL and return it as an Image object."""
    # TODO: Handle very large images like: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144225/campfire_oli_2018312_lrg.jpg
    import pillow_avif  # Keep this import, as this adds AVIF file format support to pillow
    response = requests.get(image_url, stream=True, timeout=10)
    response.raise_for_status()
    img = PillowImage.open(BytesIO(response.content))
    return Image(pillow_image=img)  # TODO: Check for duplicates


def download(url: str, max_size: int = None) -> Any:
    """Downloads a binary file from a given URL.
    TODO: Implement max file size."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
    return response.content
