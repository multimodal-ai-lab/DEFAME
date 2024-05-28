from typing import Sequence, Optional, Any
import numpy as np
import torch
from PIL import Image
from logging import Logger

from common.console import gray, light_blue, bold
from common.label import Label
from common.modeling import Model, MultimodalModel
from common import modeling_utils
from common.shared_config import path_to_data
from safe.claim_extractor import ClaimExtractor
from safe.reasoner import Reasoner
from safe.searcher import Searcher
from eval.logging import print_log


class FactChecker:
    def __init__(self,
                 model: str | Model = "OPENAI:gpt-3.5-turbo-0125",
                 multimodal_model: Optional[str] | Optional[Model] = None,
                 search_engine: str = "duckduck",
                 extract_claims: bool = True,
    ):
        if isinstance(model, str):
            model = Model(model)
        self.model = model

        if isinstance(multimodal_model, str):
            multimodal_model = MultimodalModel(multimodal_model)
        self.multimodal_model = multimodal_model

        self.claim_extractor = ClaimExtractor(model)
        self.extract_claims = extract_claims

        self.searcher = Searcher(search_engine, model)
        self.reasoner = Reasoner(model)

    def check(
            self,
            content: str | Sequence[str],
            image: Optional[torch.Tensor] = None,
            verbose: Optional[bool] = False,
            limit_search: Optional[bool] = True,
            logger: Optional[Logger] = None,
    ) -> Label:
        """
        Fact-checks the given content by first extracting all elementary claims and then
        verifying each claim individually. Returns the overall veracity which is true iff
        all elementary claims are true.
        """

        if image:
            if not self.multimodal_model:
                raise AssertionError("please specify which multimodal model to use.")
            prompt = modeling_utils.prepare_interpretation(content)
            content = self.multimodal_model.generate(image=image, prompt=prompt)
            if verbose:
                print(bold(f"Interpreting Multimodal Content:\n"))
                print(bold(light_blue(f"{content}")))
            if logger:
                print_log(logger, f"Interpreting Multimodal Content:")
                print_log(logger, f"{content}")
        if verbose:
            print(logger, bold(f"Content to be fact-checked:\n'{light_blue(content)}'"))
        if logger:
            print_log(logger, f"Content to be fact-checked:\n'{content}'")

        claims = self.claim_extractor.extract_claims(content, verbose=verbose, logger=logger) if self.extract_claims else [content]

        if verbose:
            print(bold("Verifying the claims..."))
        if logger:
            print_log(logger, "Verifying the claims...")
        veracities = []
        justifications = []
        for claim in claims:
            veracity, justification = self.verify_claim(claim, verbose=verbose, limit_search=limit_search, logger=logger)
            veracities.append(veracity)
            justifications.append(justification)

        for claim, veracity, justification in zip(claims, veracities, justifications):
            if verbose:
                print(bold(f"The claim '{light_blue(claim)}' is {veracity.value}."))
                print(gray(f'{justification}\n'))
            if logger:
                print_log(logger, f"The claim '{claim}' is {veracity.value}.")
                print_log(logger, f'{justification}')
        overall_veracity = aggregate_predictions(veracities)
        if verbose:
            print(bold(f"So, the overall veracity is: {overall_veracity.value}"))
        if logger:
            print_log(logger, f"So, the overall veracity is: {overall_veracity.value}")
        return overall_veracity

    def verify_claim(
            self, 
            claim: str, 
            verbose: Optional[bool] = False, 
            limit_search: Optional[bool] = False,
            logger: Optional[Logger] = None,
    ) -> (Label, str):
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it."""
        # TODO: Enable the model to dynamically choose the tool to use while doing
        # interleaved reasoning and evidence retrieval
        search_results = self.searcher.search(claim, verbose=verbose, limit_search=limit_search, logger=logger)
        verdict, justification = self.reasoner.reason(claim, evidence=search_results, logger=logger)
        return verdict, justification


def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    veracities = np.array(veracities)
    if np.all(veracities == Label.SUPPORTED):
        return Label.SUPPORTED
    elif np.any(veracities == Label.NEI):
        return Label.NEI
    else:
        return Label.REFUTED


def main():
    # Example usage
    model = "huggingface:meta-llama/Meta-Llama-3-70B-Instruct"
    multimodal_model = "huggingface:llava-hf/llava-1.5-7b-hf"
    fc = FactChecker(model=model, multimodal_model=multimodal_model)

    #image_url = "https://llava-vl.github.io/static/images/view.jpg"
    #image = Image.open(requests.get(image_url, stream=True).raw)
    #prompt = "This is an image of a lake in the Sahara."
    #prediction = fc.check(prompt, image)
    #print(f"Generated Prediction: {prediction}")

    print(f"Alternatively, pulling image from path:\n")

    image_path = path_to_data + "MAFC_test/image_claims/00000.png"
    image = Image.open(image_path)
    image
    prompt = "The red area has a smaller population than the US prison population."
    prediction = fc.check(prompt, image)
    print(f"Generated Prediction: {prediction}")


if __name__ == "__main__":
    main()
