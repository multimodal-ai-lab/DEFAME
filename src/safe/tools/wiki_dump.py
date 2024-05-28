import os.path
import pickle
import re
import sqlite3
import struct
from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from common.shared_config import path_to_data


class WikiDumpAPI:
    db_file_path = path_to_data + "FEVER/wiki.db"
    title_knn_path = path_to_data + "FEVER/title_knn.pckl"
    body_knn_path = path_to_data + "FEVER/body_knn.pckl"

    """Class for querying the local SQLite database from the FEVER challenge."""

    def __init__(self):
        if not os.path.exists(self.db_file_path):
            print("Warning: No FEVER database found. Continuing without database.")
            return
        self.db = sqlite3.connect(self.db_file_path, uri=True)
        self.cur = self.db.cursor()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._load_embeddings()

    def _load_embeddings(self):
        if os.path.exists(self.title_knn_path) and os.path.exists(self.body_knn_path):
            self._restore_knn_learners()
        else:
            self._build_knn_learners()

    def _restore_knn_learners(self):
        print("Restoring existing kNN learners... ", end="")
        with open(self.title_knn_path, "rb") as f:
            self.title_embeddings = pickle.load(f)
        with open(self.body_knn_path, "rb") as f:
            self.body_embeddings = pickle.load(f)
        print("done.")

    def _build_knn_learners(self):
        stmt = "SELECT ROWID, title_embedding, body_embedding FROM articles ORDER BY ROWID"
        embeddings = pd.read_sql_query(stmt, self.db)
        print("Reading title embeddings...")
        title_embeddings = df_embedding_to_np_embedding(embeddings["title_embedding"])
        print("Reading body embeddings...")
        body_embeddings = df_embedding_to_np_embedding(embeddings["body_embedding"])
        print("Setting up nearest neighbor learners...")
        self.title_embeddings = NearestNeighbors(n_neighbors=10).fit(title_embeddings)
        self.body_embeddings = NearestNeighbors(n_neighbors=10).fit(body_embeddings)
        print("Saving learners...")
        with open(self.title_knn_path, "wb") as f:
            pickle.dump(self.title_embeddings, f)
        with open(self.body_knn_path, "wb") as f:
            pickle.dump(self.body_embeddings, f)

    def search(self, phrase: str, n_results: int = 10) -> str:
        result = self.search_semantically(phrase, n_results)
        return postprocess(result)

    def _search_exact_title(self, phrase: str) -> Sequence:
        stmt = """
            SELECT *
            FROM articles
            WHERE title = ?
            LIMIT 1;
            """

        return self._run_sql_query(stmt, phrase)

    def search_semantically(self, phrase: str, n_results: int = 10) -> Sequence:
        """Performs a vector search on the text embeddings."""
        phrase_embedding = self.model.encode(phrase, show_progress_bar=False).reshape(1, -1)
        indices = self.get_nearest_neighbors_title_and_body(phrase_embedding, n_results)
        results = [self.retrieve(i) for i in indices]
        return results

    def get_nearest_neighbors_title_and_body(self, phrase_embedding, n_results: int = 10) -> Sequence[int]:
        """Returns the (deduplicated) indices of the embeddings that are closest to
        the given phrase embedding for both, the titles and the bodies."""
        n_neighbors = n_results // 2
        distances_title, indices_title = self.title_embeddings.kneighbors(phrase_embedding, n_neighbors)
        distances_body, indices_body = self.body_embeddings.kneighbors(phrase_embedding, n_neighbors)

        distances = np.hstack([distances_title, distances_body]).flatten()
        indices = np.hstack([indices_title, indices_body]).flatten()

        order_sorted = np.argsort(distances)
        return np.unique(indices[order_sorted])

    def retrieve(self, idx: int):
        """Retrieves the corresponding title and body text for the given index."""
        stmt = f"""
            SELECT title, body
            FROM articles
            WHERE ROWID = {idx + 1};
            """
        return self._run_sql_query(stmt)[0]

    def _run_sql_query(self, stmt, *args):
        self.cur.execute(stmt, args)
        rows = self.cur.fetchall()
        return rows


def preprocess(phrase: str) -> str:
    """Preprocess phrase to remove special characters"""
    phrase = re.sub(r'[^a-zA-Z0-9]', " ", phrase)
    return phrase


def postprocess(result: Sequence) -> str:
    """Keeps only the body text of the articles and concatenates them to a single string."""
    converted = ""
    if len(result) > 0:
        for row in result:
            converted += f"{process_title(row[0])}\n{process_body(row[1])}\n"
    return converted


def process_title(title: str) -> str:
    title = title.replace("_", " ")
    return process_body(title)


def process_body(body: str) -> str:
    replacement_dict = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        " ,": ",",
        " .": ".",
        " :": ":",
        " ;": ";",
        "  ": " ",
        "`` ": "\"",
        " ''": "\"",
        " '": "'",
    }
    replaced = replace(body, replacement_dict)
    replacement_dict = {
        "( ": "(",
        " )": ")",
        "[ ": "[",
        " }": "]",
        "  ": " ",
    }
    return replace(replaced, replacement_dict)


def replace(text: str, replacements: dict):
    rep = dict((re.escape(k), v) for k, v in replacements.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


def df_embedding_to_np_embedding(df: pd.DataFrame) -> np.array:
    embeddings = np.zeros(shape=(len(df), 384), dtype="float32")
    for i, embedding in enumerate(tqdm(df)):
        embeddings[i] = struct.unpack("384f", embedding)
    return embeddings
