from dataclasses import dataclass

from common.label import Label


@dataclass
class FCDoc:
    """The (incrementally growing) knowledge collection of the fact-checking (FC) process.
    Contains information like the claim that is being investigated, all intermediate reasoning
    and the evidence found. In other words, this is a protocol-like documentation of the
    fact-checking process."""

    claim: str
    protocol: list = None  # Contains intermediate reasoning and evidence
    verdict: Label = None
    justification: str = None

    def __str__(self):
        protocol_str = "\n".join([str(block) for block in self.protocol])
        return (f'Claim: "{self.claim}"\n'
                f'Fact-checking protocol: {protocol_str}')

    def add(self, block):
        self.protocol.append(block)
