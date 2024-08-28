from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Encodes text into vectors. Truncates long instances by default to 32k characters."""
    dimension: int

    def __init__(self, model_name: str, truncate_after: int = 32_000, device=None):
        self.model = SentenceTransformer(model_name,
                                         trust_remote_code=True,
                                         config_kwargs=dict(resume_download=None),
                                         device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.truncate_after = truncate_after  # num characters

    def embed(self, text: str, to_bytes: bool = False, truncate: bool = True) -> np.array:
        text = self.truncate(text) if truncate else text
        embedded = self.model.encode(text, show_progress_bar=False)
        return embedded.tobytes() if to_bytes else embedded

    def embed_many(self,
                   texts: list[str],
                   to_bytes: bool = False,
                   truncate: bool = True,
                   batch_size: int = 8) -> Sequence:
        if len(texts) == 0:
            return []
        texts = self.truncate_many(texts) if truncate else texts
        embedded = self.model.encode(texts, show_progress_bar=False, batch_size=batch_size)
        return [e.tobytes() for e in embedded] if to_bytes else embedded

    def truncate(self, text: str) -> str:
        return text[:self.truncate_after]

    def truncate_many(self, texts: list[str]) -> list[str]:
        return [self.truncate(t) for t in texts]
