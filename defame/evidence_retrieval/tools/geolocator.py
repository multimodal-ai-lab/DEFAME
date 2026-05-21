import base64
import io
from dataclasses import dataclass
from typing import Optional

import requests
from PIL.Image import Image as PILImage

from config.globals import geolocator_url
from defame.common import MultimediaSnippet, Action, Image, Results, logger
from defame.evidence_retrieval.tools.tool import Tool


class Geolocate(Action):
    """Performs geolocation to determine the country where an image was taken."""
    name = "geolocate"
    requires_image = True

    def __init__(self, image: str, top_k: int = 5):
        """
        @param image: The reference of the image to be geolocated.
        @param top_k: Number of candidates to include in the list of
            most likely countries.
        """
        self._save_parameters(locals())
        self.image = Image(reference=image)
        self.top_k = top_k

    def __eq__(self, other):
        return isinstance(other, Geolocate) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class GeolocationResults(Results):
    text: str
    most_likely_location: str
    top_k_locations: list[str]
    model_output: Optional[any] = None

    def __str__(self):
        locations_str = ', '.join(self.top_k_locations)
        text = (f'Most likely location: {self.most_likely_location}\n'
                f'Top {len(self.top_k_locations)} locations: {locations_str}')
        return text

    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None


class Geolocator(Tool):
    """Localizes a given photo by calling a remote geolocator server."""
    name = "geolocator"
    actions = [Geolocate]
    summarize = False

    def __init__(self, top_k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.server_url = geolocator_url
        logger.log(f"Initializing geolocator (server: {self.server_url})...")

    def _perform(self, action: Geolocate) -> GeolocationResults:
        return self.locate(action.image.image, top_k=action.top_k)

    def locate(self, image: PILImage, choices: list[str] | None = None, top_k: int | None = None) -> GeolocationResults:
        """
        Perform geolocation on an image via the remote geolocator server.

        :param image: A PIL image.
        :param choices: A list of location choices. If None, the server uses its default country list.
        :param top_k: Number of top results to return. Defaults to self.top_k.
        :return: A GeolocationResults object containing location predictions.
        """
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {"image_b64": image_b64, "top_k": top_k or self.top_k}
        if choices is not None:
            payload["choices"] = choices

        try:
            response = requests.post(f"{self.server_url}/geolocate", json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            logger.error(f"Geolocator server unreachable at {self.server_url}. "
                         f"Start it with: sbatch geolocator_job.sh")
            return GeolocationResults(text="Geolocator server unavailable.", most_likely_location="", top_k_locations=[])
        except Exception as e:
            logger.error(f"Geolocator request failed: {e}")
            return GeolocationResults(text=f"Geolocator error: {e}", most_likely_location="", top_k_locations=[])

        data = response.json()
        result = GeolocationResults(
            text=data["text"],
            most_likely_location=data["most_likely_location"],
            top_k_locations=data["top_k_locations"],
            model_output=True,  # non-None signals success to is_useful()
        )
        logger.log(str(result))
        return result

    def _summarize(self, result: GeolocationResults, **kwargs) -> Optional[MultimediaSnippet]:
        return MultimediaSnippet(result.text)
