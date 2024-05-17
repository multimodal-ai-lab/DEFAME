from typing import Sequence, Optional

import numpy as np

from common.console import gray, light_blue, bold
from common.label import Label
from common.modeling import Model
from safe.claim_extractor import ClaimExtractor
from safe.reasoner import Reasoner
from safe.searcher import Searcher


class FactChecker:
    def __init__(self,
                 model: str | Model = "OPENAI:gpt-3.5-turbo-0125",
                 search_tool: str = "serper",
                 extract_claims: bool = True):
        if isinstance(model, str):
            model = Model(model)
        self.model = model

        self.claim_extractor = ClaimExtractor(model)
        self.extract_claims = extract_claims

        self.searcher = Searcher(search_tool, model)
        self.reasoner = Reasoner(model)

    def check(self, content: str | Sequence[str], verbose: Optional[bool] = False) -> Label:
        """Fact-checks the given content by first extracting all elementary claims and then
        verifying each claim individually. Returns the overall veracity which is true iff
        all elementary claims are true."""

        print(bold(f"Content to be fact-checked: '{light_blue(content)}'"))

        claims = self.claim_extractor.extract_claims(content) if self.extract_claims else [content]

        print(bold("Verifying the claims..."))
        veracities = []
        justifications = []
        for claim in claims:
            veracity, justification = self.verify_claim(claim, verbose=verbose)
            veracities.append(veracity)
            justifications.append(justification)

        for claim, veracity, justification in zip(claims, veracities, justifications):
            print(bold(f"The claim '{light_blue(claim)}' is {veracity.value}."))
            print(gray(justification))
            print()

        overall_veracity = aggregate_predictions(veracities)
        print(bold(f"So, the overall veracity is: {overall_veracity.value}"))

        return overall_veracity

    def verify_claim(self, claim: str, verbose: Optional[bool] = False) -> (Label, str):
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it."""
        # TODO: Enable the model to dynamically choose the tool to use
        # TODO: Enable interleaved reasoning and evidence retrieval
        search_results = self.searcher.search(claim, verbose)
        verdict, justification = self.reasoner.reason(claim, evidence=search_results)
        return verdict, justification
    
def aggregate_predictions(veracities: Sequence[Label]) -> Label:
        overall_supported = np.all(np.array(veracities) == Label.SUPPORTED)
        overall_veracity = Label.SUPPORTED if overall_supported else Label.REFUTED
        return overall_veracity
