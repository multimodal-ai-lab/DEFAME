from typing import Collection
from dataclasses import dataclass

from common.label import Label
from common.action import Action
from common.results import Result


@dataclass
class ReasoningBlock:
    text: str

    def __str__(self):
        return f"REASONING:\n{self.text}"


@dataclass
class ActionsBlock:
    actions: list[Action]

    def __str__(self):
        actions_str = "\n".join([str(a) for a in self.actions])
        return f"ACTIONS:\n```\n{actions_str}\n```"


@dataclass
class ResultsBlock:
    results: Collection[Result]

    def __str__(self):
        if len(self.results) > 0:
            results_str = "\n".join([str(r) for r in self.results if r.is_useful()])
        else:
            results_str = "No new results found!"
        return f"RESULTS:\n{results_str}"


class FCDoc:
    """The (incrementally growing) knowledge collection of the fact-checking (FC) process.
    Contains information like the claim that is being investigated, all intermediate reasoning
    and the evidence found. In other words, this is a protocol-like documentation of the
    fact-checking process."""

    claim: str
    protocol: list  # Contains intermediate reasoning and evidence
    verdict: Label = None
    justification: str = None

    def __init__(self, claim: str):
        self.claim = claim
        self.protocol = []

    def __str__(self):
        doc_str = f'CLAIM:\n"{self.claim}"'
        if self.protocol:
            doc_str += "\n\n" + "\n\n".join([str(block) for block in self.protocol])
        if self.verdict:
            doc_str += f"\n\nVERDICT: {self.verdict.name}"
        if self.justification:
            doc_str += f"\n\nJUSTIFICATION:\n{self.justification}"
        return doc_str

    def add_reasoning(self, text: str):
        self.protocol.append(ReasoningBlock(text))

    def add_actions(self, actions: list[Action]):
        self.protocol.append(ActionsBlock(actions))

    def add_results(self, results: Collection[Result]):
        self.protocol.append(ResultsBlock(results))

    def get_all_reasoning(self) -> list[str]:
        reasoning_texts = []
        for block in self.protocol:
            if isinstance(block, ReasoningBlock):
                reasoning_texts.append(block.text)
        return reasoning_texts
