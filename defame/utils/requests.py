from io import BytesIO
from typing import Optional, Any

import requests
from PIL import Image as pillowImage

from defame.common import Image


def download_image(image_url: str) -> Optional[Image]:
    """Download an image from a URL and return it as an Image object."""
    response = requests.get(image_url, timeout=7)
    response.raise_for_status()
    return Image(pillow_image=pillowImage.open(BytesIO(response.content)))


def download(url: str, max_size: int = None) -> Any:
    """Downloads a binary file from a given URL.
    TODO: Implement max file size."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
    return response.content
