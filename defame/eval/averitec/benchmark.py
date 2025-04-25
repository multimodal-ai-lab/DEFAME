import json
from datetime import datetime

from defame.common import Label, Claim, logger
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Search


class AVeriTeC(Benchmark):
    name = "AVeriTeC"
    shorthand = "averitec"

    is_multimodal = False

    class_mapping = {
        "Supported": Label.SUPPORTED,
        "Not Enough Evidence": Label.NEI,
        "Conflicting Evidence/Cherrypicking": Label.CONFLICTING,
        "Refuted": Label.REFUTED,
    }

    class_definitions = {
        Label.SUPPORTED:
            """The knowledge from the fact-check supports or at least strongly implies the Claim.
            Mere plausibility is not enough for this decision.""",
        Label.NEI:
            """The fact-check does not contain sufficient information to come to a conclusion. In particular,
            there is substantial lack of both supporting and refuting evidence.""",
        Label.CONFLICTING:
            """The Claim has both supporting and refuting evidence from multiple sources or the Claim is technically true but misleads by excluding important context. """,
        Label.REFUTED:
            """The knowledge from the fact-check explicitly and clearly refutes at least substantial parts
            if not even the whole the Claim.""",
    }

    extra_judge_rules = """* **Do not commit the "argument from ignorance" fallacy**: The absence of evidence
    for the Claim does NOT prove that the Claim is refuted. Instead, the Claim is simply unsupported--which is a case of 
    'not enough information'."""

    available_actions = [Search]

    def __init__(self, variant: str = "dev"):
        super().__init__(variant=variant, file_path=f"AVeriTeC/{variant}.json")

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            data_raw = json.load(f)

        data = []
        for i, d in enumerate(data_raw):
            date = d["claim_date"]
            identifier = str(i)
            claim = Claim(
                d["claim"],
                author=d["speaker"],
                date=datetime.strptime(date, "%d-%m-%Y") if date else None,
                origin=d["original_claim_url"],
                id=identifier
            )
            label = self.class_mapping[d["label"]] if self.variant in ["train", "dev"] else None
            justification = d["justification"] if self.variant in ["train", "dev"] else None

            data.append({"id": identifier, "input": claim, "label": label, "justification": justification})

        return data

    def process_output(self, output):
        doc, meta = output
        claim = doc.claim
        instance = self.get_by_id(claim.id)
        prediction = doc.verdict

        # Special output processing for AVeriTeC
        if prediction == Label.CHERRY_PICKING:
            # Merge cherry-picking and conflicting label
            prediction = Label.CONFLICTING

        pred_label = self.get_class_name(prediction)
        averitec_out_instance = {
            "claim_id": claim.id,
            "claim": str(claim).replace("\n", " "),
            "pred_label": pred_label
        }

        if "q_and_a" in meta:
            averitec_out_instance["evidence"] = meta["q_and_a"]

        logger.save_next_averitec_out(averitec_out_instance)

        self._save_prediction(doc, meta, claim, prediction, instance.get("label"), instance.get("justification"))
