from dataclasses import dataclass
from typing import Collection
from pathlib import Path
import shutil

import numpy as np
from ezmm import MultimodalSequence
from markdown_pdf import MarkdownPdf, Section

from defame.common import Action, Claim, Label, Evidence
from defame.utils.parsing import replace_item_refs


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


class Report:
    """An (incrementally growing) document, recording the fact-check. It contains
    information like the claim, retrieved evidence, and all intermediate reasoning."""

    claim: Claim
    record: list  # contains intermediate reasoning and evidence, organized in blocks
    verdict: Label = None
    justification: str = None

    def __init__(self, claim: Claim):
        self.claim = claim
        self.record = []
        # if claim.original_context.interpretation:
        #     self.add_reasoning("## Interpretation\n" + claim.original_context.interpretation)

    def save_to(self, directory: str | Path):
        """Saves the Report to the specified directory. Exports the raw
        Markdown report, all referenced media files, and a rendered PDF."""
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        report_str = str(self)
        seq = MultimodalSequence(report_str)
        media = seq.unique_items()
        report_str = replace_item_refs(report_str, media)

        # Save the Markdown file
        with open(directory / "report.md", "w") as f:
            f.write(report_str)

        # Save all associated media files in a separate subdirectory
        media_dir = directory / "media"
        media_dir.mkdir(exist_ok=True)
        for medium in media:
            medium_copy_path = media_dir / medium.file_path.name
            shutil.copy(medium.file_path, medium_copy_path)

        # Save a rendered PDF
        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(Section(report_str, toc=False, root=directory.as_posix()))

        pdf.meta["title"] = "Fact-Check Report"
        pdf.save(directory / "report.pdf")

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

    def get_result_as_dict(self) -> dict:
        """Returns the final verdict and the justification as a dictionary."""
        return {"verdict": self.verdict.name, "justification": self.justification}
