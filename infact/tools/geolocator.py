from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from infact.common.action import Geolocate
from infact.common.results import GeolocationResult, Result
from infact.common.logger import Logger
from infact.tools.tool import Tool


class Geolocator(Tool):
    """Localizes a given photo."""
    name = "geolocator"
    actions = [Geolocate]

    def __init__(self,
                 model_name: str = "geolocal/StreetCLIP",
                 top_k=10,
                 logger: Logger = None, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the GeoLocator with a pretrained model from Hugging Face.

        :param model_name: The name of the Hugging Face model to use for geolocation.
        :param device: The device to run the model on (e.g., -1 for CPU, 0 for GPU).
        :param use_multiple_gpus: Whether to use multiple GPUs if available.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.top_k = top_k

        if use_multiple_gpus and torch.cuda.device_count() > 1:  # TODO
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

    def perform(self, action: Geolocate) -> list[Result]:
        return [self.locate(action.image)]

    def locate(self, image: Image.Image, choices: List[str] = None) -> GeolocationResult:
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
        confidences = {choices[i]: float(prediction[0][i].item()) for i in range(len(choices))}
        top_k_locations = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:self.top_k])
        most_likely_location = max(top_k_locations, key=top_k_locations.get)
        model_output = logits_per_image
        result = GeolocationResult(
            source=self.model_name,
            text=f"The most likely countries where the image was taken are: {top_k_locations}",
            most_likely_location=most_likely_location,
            top_k_locations=top_k_locations,
            model_output=model_output
        )
        self.logger.log(str(result))
        return result
