from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Encodes text into vectors."""
    dimension = 768

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, text: str, to_bytes: bool = False) -> np.array:
        embedded = self.model.encode(text, show_progress_bar=False)
        return embedded.tobytes() if to_bytes else embedded

    def embed_many(self, texts: Sequence[str], to_bytes: bool = False) -> Sequence:
        return [self.embed(t, to_bytes=to_bytes) for t in texts]
