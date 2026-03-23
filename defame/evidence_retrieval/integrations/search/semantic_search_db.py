import os
import pickle
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from ezmm import MultimodalSequence
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from config.globals import embedding_model
from defame.common.embedding import EmbeddingModel
from defame.evidence_retrieval.integrations.search.local_search_platform import LocalSearchPlatform
from .common import SearchResults, Query, WebSource


class SemanticSearchDB(LocalSearchPlatform):
    def __init__(self, db_file_path: str | Path):
        super().__init__()
        self.is_free = True
        self.db_file_path = db_file_path
        self.embedding_model = None
        if not os.path.exists(self.db_file_path):
            print(f"Warning: No {self.name} database found at '{self.db_file_path}'. Creating new one.")
        os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
        self.db = sqlite3.connect(self.db_file_path, uri=True)
        self.cur = self.db.cursor()

    def is_empty(self) -> bool:
        """Returns True iff the database is empty."""
        raise NotImplementedError

    def _embed(self, *args, **kwargs):
        if self.embedding_model is None:
            self._setup_embedding_model()
        return self.embedding_model.embed(*args, **kwargs)

    def _embed_many(self, *args, **kwargs):
        if self.embedding_model is None:
            self._setup_embedding_model()
        return self.embedding_model.embed_many(*args, **kwargs)

    def _setup_embedding_model(self):
        self.embedding_model = EmbeddingModel(embedding_model)

    def _restore_knn_from(self, path: str) -> NearestNeighbors:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _run_sql_query(self, stmt: str, *args) -> Sequence:
        """Runs the SQL statement stmt (with optional arguments) on the DB and returns the rows."""
        self.cur.execute(stmt, args)
        rows = self.cur.fetchall()
        return rows

    def _call_api(self, query: Query) -> Optional[SearchResults]:
        query_embedding = self._embed(query).reshape(1, -1)
        indices = self._search_semantically(query_embedding, query.limit)
        web_sources = self._indices_to_search_results(indices)
        return SearchResults(sources=web_sources, query=query)

    def _search_semantically(self, query_embedding, limit: int) -> list[int]:
        """Runs a semantic search using kNN. Returns the indices (starting at 0)
        of the search results."""
        raise NotImplementedError()

    def retrieve(self, idx: int) -> (str, str, datetime):
        """Selects the row with specified index from the DB and returns the URL, the text
        and the date of the selected row's source."""
        raise NotImplementedError()

    def _indices_to_search_results(self, indices: list[int]) -> list[WebSource]:
        results = []
        for i, index in enumerate(indices):
            url, text, date = self.retrieve(index)
            result = WebSource(
                reference=url,
                content=MultimodalSequence(text),
                release_date=date
            )
            results.append(result)
        return results

    def _build_db(self, **kwargs) -> None:
        """Creates the SQLite database."""
        raise NotImplementedError()


def df_embedding_to_np_embedding(df: pd.DataFrame, dimension: int) -> np.array:
    """Converts a Pandas DataFrame of binary embeddings into the respective
    NumPy array with shape (num_instances, dimension) containing the unpacked embeddings."""
    embeddings = np.zeros(shape=(len(df), dimension), dtype="float32")
    for i, embedding in enumerate(tqdm(df)):
        if embedding is not None:
            embeddings[i] = struct.unpack(f"{dimension}f", embedding)
        else:
            embeddings[i] = 1000  # Put invalid "embeddings" far away
    return embeddings
