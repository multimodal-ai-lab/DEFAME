from typing import Sequence

import numpy as np
import torch
from ezmm import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


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

class MultimodalEmbeddingModel:

    def __init__(self, model_name: str, device="auto"):
        self.model = AutoModel.from_pretrained(model_name, device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_name, device_map=device)

    # todo better parallelization
    def embed(self, content: str | Image, to_bytes: bool = False) -> np.array:
        with torch.no_grad():
            if isinstance(content, Image):
                inp = self.processor(images=[content.get_base64_encoded()], return_tensors="pt")
                inp = {k: v.to(self.model.device) for k, v in inp.items()}
                embedding = self.model.get_image_features(**inp)
            else:
                inp = self.processor(text=[content], return_tensors="pt", padding=True)
                inp = {k: v.to(self.model.device) for k, v in inp.items()}
                embedding = self.model.get_text_features(**inp)

        res =  embedding.cpu().numpy()
        return res.tobytes() if to_bytes else res

    def embed_many(self,
                   content: list[str | Image],
                   to_bytes: bool = False,
                   batch_size: int = 8) -> Sequence:

        if len(content) == 0:
            return []

        output = []
        for c in content:
            output.append(self.embed(content=c, to_bytes=to_bytes))
        return output


def similarity(embedding1: np.array, embedding2: np.array) -> float:
    return np.dot(embedding1.ravel(), embedding2.ravel()) / (np.linalg.norm(embedding1.ravel()) * np.linalg.norm(embedding2.ravel()))

