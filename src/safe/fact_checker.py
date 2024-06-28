from typing import Sequence, Optional, Collection

import numpy as np
from PIL import Image

from common import modeling_utils
from common.console import gray, light_blue, bold
from common.label import Label
from common.modeling import Model, MultimodalModel
from common.shared_config import path_to_data
from common.claim import Claim
from common.results import Result
from eval.logger import EvaluationLogger
from safe.claim_extractor import ClaimExtractor
from safe.judge import Judge
from safe.prompts.prompt import PreparePrompt, ReiteratePrompt
from common.document import FCDocument
from safe.planner import Planner
from safe.actor import Actor
from safe.doc_summarizer import DocSummarizer
from safe.result_summarizer import ResultSummarizer
from common.action import *


class FactChecker:
    def __init__(self,
                 multimodal: bool = False,
                 model: str | Model = "OPENAI:gpt-3.5-turbo-0125",
                 multimodal_model: Optional[str] | Optional[Model] = None,
                 search_engines: list[str] = None,
                 extract_claims: bool = False,
                 max_iterations: int = 5,
                 max_results_per_search: int = 3,
                 logger: EvaluationLogger = None,
                 classes: Sequence[Label] = None,
                 class_definitions: dict[Label, str] = None,
                 extra_prepare_rules: str = None,
                 extra_plan_rules: str = None,
                 extra_judge_rules: str = None,
                 verbose: bool = True,
                 ):
        self.logger = logger or EvaluationLogger(verbose=verbose)
        
        if isinstance(model, str):
            model = Model(model, logger)
        self.model = model

        if isinstance(multimodal_model, str):
            multimodal_model = MultimodalModel(multimodal_model, logger)
        self.multimodal_model = multimodal_model

        self.claim_extractor = ClaimExtractor(model, self.logger)
        self.extract_claims = extract_claims

        if classes is None:
            if class_definitions is None:
                classes = [Label.SUPPORTED, Label.NEI, Label.REFUTED]
            else:
                classes = list(class_definitions.keys())

        # TODO: add to parameters   
        actions = []
        if search_engines:
            if "wiki_dump" in search_engines:
                actions.append(WikiDumpLookup)
            if "google" in search_engines or "duckduckgo" in search_engines or "averitec_kb" in search_engines:
                actions.append(WebSearch)
        if multimodal:
            actions.append(ObjectRecognition)
            #actions.append(ReverseSearch)
            actions.append(GeoLocation)
            #actions.append(FaceRecognition)
            actions.append(SourceCredibilityCheck)
            actions.append(OCR)

        self.planner = Planner(multimodal=multimodal, 
                               valid_actions=actions, 
                               model=self.model, 
                               logger=self.logger, 
                               extra_rules=extra_plan_rules)
        self.actor = Actor(multimodal=multimodal,
                           model=self.model, 
                           search_engines=search_engines, 
                           max_results_per_search=max_results_per_search, 
                           logger=self.logger)
        self.judge = Judge(multimodal=multimodal,
                           model=self.model, 
                           logger=self.logger, 
                           classes=classes, 
                           class_definitions=class_definitions, 
                           extra_rules=extra_judge_rules)
        self.doc_summarizer = DocSummarizer(self.model, self.logger)
        self.result_summarizer = ResultSummarizer(self.model, self.logger)

        self.extra_prepare_rules = extra_prepare_rules
        self.max_iterations = max_iterations

    def check(
            self,
            content: Claim | str | Sequence[str],
            images: Optional[list[Image.Image]] = None,
    ) -> FCDocument:
        """
        Fact-checks the given content by first extracting all elementary claims and then
        verifying each claim individually. Returns the overall veracity which is true iff
        all elementary claims are true.
        """
        # TODO: rework this method
        if images:
            if not self.multimodal_model:
                raise AssertionError("please specify which multimodal model to use.")
            prompt = modeling_utils.prepare_interpretation(content)
            #maybe it has to be content instead of multimodal_content
            multimodal_content = self.multimodal_model.generate(image=image, prompt=prompt)
            self.logger.log(bold(f"Interpreting Multimodal Content:\n"))
            self.logger.log(bold(light_blue(f"{multimodal_content}")))

        self.logger.log(bold(f"Content to be fact-checked:\n'{light_blue(content.text)}'"), important=True)
        claims = self.claim_extractor.extract_claims(content) if self.extract_claims else [content]

        self.logger.log(bold("Verifying the claims..."), important=True)
        docs = []
        for claim in claims:
            doc = self.verify_claim(claim, images)
            docs.append(doc)
            self.logger.log(bold(f"The claim '{light_blue(str(claim.text))}' is {doc.verdict.value}."), important=True)
            if doc.justification:
                self.logger.log(f'Justification: {gray(doc.justification)}', important=True)

        overall_veracity = aggregate_predictions([doc.verdict for doc in docs])
        self.logger.log(bold(f"So, the overall veracity is: {overall_veracity.value}"))
        return doc

    def verify_claim(self, claim: Claim, images: Optional[list[Image.Image]] = None) -> FCDocument:
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        self.actor.searcher.reset()  # remove all past search results
        doc = FCDocument(claim)
        self._prepare_fact_check(doc)
        label = Label.NEI
        n_iterations = 0
        while True:
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc) # TODO: Missing Grounding Maybe because it doesn't actually see the Image. Still using LLM instead of MLLM
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            doc.add_actions(actions)
            if not actions:
                break  # the planner wasn't able to determine further useful actions, giving up
            results = self.actor.perform(actions, images)
            results = self.result_summarizer.summarize(results, doc)
            doc.add_results(results)  # even if no results, add empty results block for the record
            self._consolidate_new_knowledge(doc, results)
            label = self.judge.judge(doc)
            if label != Label.NEI or n_iterations == self.max_iterations:
                break
            else:
                self.logger.log("Not enough information yet. Continuing fact-check...")
        doc.add_reasoning(self.judge.get_latest_reasoning())
        doc.verdict = label
        if label != Label.REFUSED_TO_ANSWER:
            doc.justification = self.doc_summarizer.summarize(doc)
        return doc

    def _prepare_fact_check(self, doc: FCDocument):
        """Does some initial reasoning (like claim interpretation and question generation)."""
        prompt = PreparePrompt(doc.claim, self.extra_prepare_rules)
        answer = self.model.generate(str(prompt))
        doc.add_reasoning(answer)

    def _consolidate_new_knowledge(self, doc: FCDocument, results: Collection[Result]):
        """Analyzes the currently available information and states new questions, adds them
        to the FCDoc."""
        prompt = ReiteratePrompt(doc, results)
        answer = self.model.generate(str(prompt))
        doc.add_reasoning(answer)


def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    veracities = np.array(veracities)
    if np.all(veracities == Label.SUPPORTED):
        return Label.SUPPORTED
    elif np.any(veracities == Label.REFUTED):
        return Label.REFUTED
    elif np.any(veracities == Label.REFUSED_TO_ANSWER):
        return Label.REFUSED_TO_ANSWER
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
    images = [Image.open(image_path)]
    images[0]
    prompt = "The red area has a smaller population than the US prison population."
    prediction = fc.check(prompt, images)
    print(f"Generated Prediction: {prediction}")


if __name__ == "__main__":
    main()
