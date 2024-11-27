from dataclasses import dataclass
from typing import Collection

import numpy as np

from defame.common.action import Action
from defame.common.claim import Claim
from defame.common.label import Label
from defame.common.evidence import Evidence


@dataclass
class ReasoningBlock:
    text: str

    def __str__(self):
        return self.text if self.text else "None"


@dataclass
class ActionsBlock:
    actions: list[Action]

    def __str__(self):
        actions_str = "\n".join([str(a) for a in self.actions])
        return f"## Actions\n```\n{actions_str}\n```"


@dataclass
class EvidenceBlock:
    evidences: Collection[Evidence]

    def __str__(self):
        any_is_useful = np.any([e.is_useful() for e in self.evidences])
        if not any_is_useful:
            summary = "No new evidence found."
        else:
            summary = "\n\n".join([str(e) for e in self.evidences if e.is_useful()])
        return f"## Evidence\n{summary}"

    @property
    def num_useful_evidences(self):
        n_useful = 0
        for e in self.evidences:
            if e.is_useful():
                n_useful += 1
        return n_useful

    def get_useful_evidences_str(self) -> str:
        if self.num_useful_evidences > 0:
            useful_evidences = [str(e) for e in self.evidences if e.is_useful()]
            return "\n\n".join(useful_evidences)
        else:
            return "No new useful evidence!"


class FCDocument:
    """An (incrementally growing) record, documenting the fact-checking (FC) process.
    Contains information like the claim that is being investigated, all intermediate reasoning
    and the evidence found."""

    claim: Claim
    record: list  # contains intermediate reasoning and evidence, organized in blocks
    verdict: Label = None
    justification: str = None

    def __init__(self, claim: Claim):
        self.claim = claim
        self.record = []
        if claim.original_context.interpretation:
            self.add_reasoning("## Interpretation\n" + claim.original_context.interpretation)

    def __str__(self):
        doc_str = f'## Claim\n{self.claim}'
        if self.record:
            doc_str += "\n\n" + "\n\n".join([str(block) for block in self.record])
        if self.verdict:
            doc_str += f"\n\n### Verdict: {self.verdict.name}"
        if self.justification:
            doc_str += f"\n\n### Justification\n{self.justification}"
        return doc_str

    def add_reasoning(self, text: str):
        self.record.append(ReasoningBlock(text))

    def add_actions(self, actions: list[Action]):
        self.record.append(ActionsBlock(actions))

    def add_evidence(self, evidences: Collection[Evidence]):
        self.record.append(EvidenceBlock(evidences))

    def get_all_reasoning(self) -> list[str]:
        reasoning_texts = []
        for block in self.record:
            if isinstance(block, ReasoningBlock):
                reasoning_texts.append(block.text)
        return reasoning_texts

    def get_all_actions(self) -> list[Action]:
        all_actions = []
        for block in self.record:
            if isinstance(block, ActionsBlock):
                all_actions.extend(block.actions)
        return all_actions
