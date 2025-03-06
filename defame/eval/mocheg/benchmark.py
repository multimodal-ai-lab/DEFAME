from typing import Iterator
import pandas as pd

from config.globals import data_root_dir
from defame.common import Label, Content
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import WebSearch, ImageSearch, ReverseSearch, Geolocate

class MOCHEG(Benchmark):
    shorthand = "mocheg"
    is_multimodal = True

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

    available_actions = [WebSearch, ImageSearch, ReverseSearch, Geolocate]

    def __init__(self, variant="val"):
        super().__init__(f"MOCHEG ({variant})", variant)
        self.file_path = data_root_dir / f"MOCHEG/{variant}/Corpus2.csv"
        if not self.file_path.exists():
            raise ValueError(f"Unable to locate MOCHEG at {data_root_dir.as_posix()}. "
                             f"See README.md for setup instructions.")

        self.image_path = data_root_dir / f"MOCHEG/{variant}/images/"
        self.data = self.load_data()

    def load_data(self) -> list[dict]:
        # Load the corpus
        df = pd.read_csv(self.file_path)
        # Remove duplicates based on 'claim_id', keeping only the first occurrence
        df = df.drop_duplicates(subset='claim_id', keep='first')
        # Remove samples with an invalid justification
        df = df.dropna(subset=['ruling_outline'])
        #df = df[df['ruling_outline'].str.split().str.len() >= 10]

        #claims = [
    #"A photo depicting snow in the Sahara desert in 2024 was widely shared, but fact-checkers say it's an older image from 2018.",
    #]
        #df = df[:len(claims)]  
        #df['Claim'] = claims 
        
        

        data = []
        for i, row in df.iterrows():
            #image_file = f"{row['claim_id']}-proof-{row['evidence_id']}.jpg" # this is not correct yet
            #image_path = self.image_path / image_file


            # Load the image evidence
            #if os.path.exists(image_path):
            #    image = Image(image_path)
            #else:
            #    image = None  # Or handle missing images
            image = None

            claim_text = row["Claim"]
            text_evidence = row["Evidence"]
            label = self.class_mapping[row["cleaned_truthfulness"].lower()]
            identifier = str(row["claim_id"])

            # Create an entry for each claim
            entry = {
                "id": identifier,
                "content": Content(data=claim_text, identifier=identifier),
                "label": label,
                "justification": row.get("ruling_outline", "")
            }
            data.append(entry)

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
