from typing import Sequence, Optional

import numpy as np
import torch
from PIL import Image

from common import modeling_utils
from common.console import gray, light_blue, bold
from common.label import Label
from common.modeling import Model, MultimodalModel
from common.shared_config import path_to_data
from eval.logger import EvaluationLogger
from safe.claim_extractor import ClaimExtractor
from safe.judge import Judge
from safe.searcher import Searcher
from common.document import FCDoc
from safe.planner import Planner
from safe.actor import Actor
from safe.doc_summarizer import DocSummarizer


class FactChecker:
    def __init__(self,
                 model: str | Model = "OPENAI:gpt-3.5-turbo-0125",
                 multimodal_model: Optional[str] | Optional[Model] = None,
                 search_engines: list[str] = None,
                 extract_claims: bool = True,
                 summarize_search_results: bool = True,
                 max_searches_per_claim: int = 5,
                 logger: EvaluationLogger = None,
                 classes: Sequence[Label] = None,
                 ):
        if isinstance(model, str):
            model = Model(model)
        self.model = model

        if isinstance(multimodal_model, str):
            multimodal_model = MultimodalModel(multimodal_model)
        self.multimodal_model = multimodal_model

        self.logger = logger or EvaluationLogger()

        self.claim_extractor = ClaimExtractor(model, logger)
        self.extract_claims = extract_claims

        if classes is None:
            classes = [Label.SUPPORTED, Label.NEI, Label.REFUTED]

        search_engines = search_engines or ["duckduck"]

        self.planner = Planner(model, self.logger)
        self.actor = Actor()
        self.searcher = Searcher(search_engines, model, self.logger,
                                 summarize_search_results, max_searches_per_claim)
        self.judge = Judge(model, self.logger, classes)
        self.doc_summarizer = DocSummarizer(model, self.logger)

    def check(
            self,
            content: str | Sequence[str],
            image: Optional[torch.Tensor] = None,
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
            self.logger.log(bold(f"Interpreting Multimodal Content:\n"))
            self.logger.log(bold(light_blue(f"{content}")))

        self.logger.log(bold(f"Content to be fact-checked:\n'{light_blue(content)}'"))

        claims = self.claim_extractor.extract_claims(content) if self.extract_claims else [content]

        self.logger.log(bold("Verifying the claims..."))
        veracities = []
        justifications = []
        evidence_log = dict()
        for claim in claims:
            veracity, justification = self.verify_claim(claim, evidence_log)
            veracities.append(veracity)
            justifications.append(justification)

        for claim, veracity, justification in zip(claims, veracities, justifications):
            self.logger.log(bold(f"The claim '{light_blue(claim)}' is {veracity.value}."))
            self.logger.log(gray(f'{justification}\n'))
        overall_veracity = aggregate_predictions(veracities)

        self.logger.log(bold(f"So, the overall veracity is: {overall_veracity.value}"))
        return evidence_log, overall_veracity

    def verify_claim(self, claim: str) -> FCDoc:
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it."""
        doc = FCDoc(claim)
        while (label := self.judge.judge(doc)) == Label.NEI:
            actions, reasoning = self.planner.plan_next_actions(doc)
            doc.add(reasoning)
            results = self.actor.perform(actions)
            doc.add(results)
        doc.verdict = label
        doc.justification = self.doc_summarizer.summarize(doc)
        return doc


def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    veracities = np.array(veracities)
    if np.all(veracities == Label.SUPPORTED):
        return Label.SUPPORTED
    elif np.any(veracities == Label.REFUTED):
        return Label.REFUTED
    elif np.any(veracities == Label.CONFLICTING):
        return Label.CONFLICTING
    else:
        return Label.NEI


def main():
    # Example usage
    model = "huggingface:meta-llama/Meta-Llama-3-70B-Instruct"
    multimodal_model = "huggingface:llava-hf/llava-1.5-7b-hf"
    fc = FactChecker(model=model, multimodal_model=multimodal_model)

    # image_url = "https://llava-vl.github.io/static/images/view.jpg"
    # image = Image.open(requests.get(image_url, stream=True).raw)
    # prompt = "This is an image of a lake in the Sahara."
    # prediction = fc.check(prompt, image)
    # print(f"Generated Prediction: {prediction}")

    print(f"Alternatively, pulling image from path:\n")

    image_path = path_to_data + "MAFC_test/image_claims/00000.png"
    image = Image.open(image_path)
    image
    prompt = "The red area has a smaller population than the US prison population."
    prediction = fc.check(prompt, image)
    print(f"Generated Prediction: {prediction}")


if __name__ == "__main__":
    main()
