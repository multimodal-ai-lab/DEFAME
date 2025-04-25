from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL.Image import Image as PILImage
from ezmm import MultimodalSequence, Image
from transformers import AutoProcessor, AutoModel

from defame.common import Action, Results, logger
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
    """Localizes a given photo."""
    name = "geolocator"
    actions = [Geolocate]
    summarize = False

    def __init__(self, model_name: str = "geolocal/StreetCLIP", top_k=10, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the GeoLocator with a pretrained model from Hugging Face.

        :param model_name: The name of the Hugging Face model to use for geolocation.
        :param device: The device to run the model on (e.g., -1 for CPU, 0 for GPU).
        :param use_multiple_gpus: Whether to use multiple GPUs if available.
        """
        # FIXME: Print warning if no GPU available
        logger.log("Initializing geolocator...")
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.top_k = top_k

        self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model.to(self.device)

    def _perform(self, action: Geolocate) -> Results:
        return self.locate(action.image.image)

    def locate(self, image: PILImage, choices: List[str] = None) -> GeolocationResults:
        """
        Perform geolocation on an image.

        :param image: A PIL image.
        :param choices: A list of location choices. If None, uses a default list of countries.
        :return: A GeoLocationResult object containing location predictions and their probabilities.
        """
        if choices is None:
            choices = ['Albania', 'Andorra', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bermuda',
                       'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China',
                       'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Estonia',
                       'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hungary',
                       'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kenya',
                       'Kyrgyzstan', 'Laos', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar',
                       'Malaysia', 'Malta', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Netherlands', 'New Zealand',
                       'Nigeria', 'Norway', 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal',
                       'Puerto Rico', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia',
                       'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden',
                       'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine',
                       'United Arab Emirates',
                       'United Kingdom', 'United States', 'Uruguay']

        inputs = self.processor(text=choices, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        prediction = logits_per_image.softmax(dim=1)

        # Compute classification score for each country
        confidences = {choices[i]: round(float(prediction[0][i].item()), 2) for i in range(len(choices))}
        top_k_locations = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:self.top_k])
        most_likely_location = max(top_k_locations, key=top_k_locations.get)
        model_output = logits_per_image
        result = GeolocationResults(
            text=f"The most likely countries where the image was taken are: {top_k_locations}",
            most_likely_location=most_likely_location,
            top_k_locations=list(top_k_locations.keys()),
            model_output=model_output
        )
        logger.log(str(result))
        return result

    def _summarize(self, result: GeolocationResults, **kwargs) -> Optional[MultimodalSequence]:
        return MultimodalSequence(result.text)  # TODO: Improve summary w.r.t. uncertainty
