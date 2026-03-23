import pandas as pd

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Search, Geolocate


class MOCHEG(Benchmark):
    name = "MOCHEG"
    shorthand = "mocheg"

    class_mapping = {
        "supported": Label.SUPPORTED,
        "nei": Label.NEI,
        "refuted": Label.REFUTED,
    }

    class_definitions = {
        Label.SUPPORTED:
            "A claim is considered supported when the provided evidence backs up the claim.",
        Label.NEI:
            "A claim is marked as NEI when there isn't enough evidence to support or refute the claim.",
        Label.REFUTED:
            "A claim is considered refuted when the evidence contradicts the claim.",
    }

    extra_judge_rules = """* **Do not commit the "argument from ignorance" fallacy**: The absence of evidence
    for the Claim does NOT prove that the Claim is refuted. Instead, the Claim is simply unsupported--which is a case of 
    'not enough information'."""

    available_actions = [Search, Geolocate]

    def __init__(self, variant="val"):
        super().__init__(variant, f"MOCHEG/{variant}/Corpus2.csv")
        self.image_path = data_root_dir / f"MOCHEG/{variant}/images/"

    def _load_data(self) -> list[dict]:
        # Load the corpus
        df = pd.read_csv(self.file_path)

        # Remove duplicates based on 'claim_id', keeping only the first occurrence
        df = df.drop_duplicates(subset='claim_id', keep='first')

        # Remove samples with an invalid justification
        df = df.dropna(subset=['ruling_outline'])

        data = []
        for i, row in df.iterrows():
            claim_text = row["Claim"]
            label = self.class_mapping[row["cleaned_truthfulness"].lower()]
            identifier = str(row["claim_id"])

            # Create an entry for each claim
            entry = {
                "id": identifier,
                "input": Claim(claim_text, id=identifier),
                "label": label,
                "justification": row.get("ruling_outline", "")
            }
            data.append(entry)

        return data
