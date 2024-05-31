import json
import os.path
import pickle
import sqlite3
import struct
from threading import Thread
from queue import Queue
from typing import Sequence

import numpy as np
import pandas as pd
import unicodedata
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from common.embedding import EmbeddingModel
from common.shared_config import path_to_data, embedding_model


class KnowledgeBase:
    """The AVeriTeC knowledge base consisting of a selection of webpages."""
    db_file_path = path_to_data + "AVeriTeC/knowledge_base.db"
    embedding_knn_path = path_to_data + "AVeriTeC/embedding_knn.pckl"

    def __init__(self):
        if not os.path.exists(self.db_file_path):
            print("Warning: No AVeriTeC database found. Continuing with empty database.")
        os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
        self.db = sqlite3.connect(self.db_file_path, uri=True)
        self.cur = self.db.cursor()
        self.embedding_model = None
        if os.path.exists(self.embedding_knn_path):
            self._restore_knn_learners()

    def _restore_knn_learners(self):
        print("Restoring existing kNN learners... ", end="")
        with open(self.embedding_knn_path, "rb") as f:
            self.embeddings = pickle.load(f)
        print("done.")

    def build_knn_learners(self):
        self._setup_embedding_model()
        stmt = "SELECT ROWID, text_embedding FROM websites ORDER BY ROWID"
        embeddings = pd.read_sql_query(stmt, self.db)
        print("Reading embeddings...")
        embeddings = df_embedding_to_np_embedding(embeddings["text_embedding"], self.embedding_model.dimension)
        print("Setting up nearest neighbor learners...")
        self.embeddings = NearestNeighbors(n_neighbors=10).fit(embeddings)
        print("Saving learners...")
        with open(self.embedding_knn_path, "wb") as f:
            pickle.dump(self.embeddings, f)

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
        phrase_embedding = self._embed(phrase).reshape(1, -1)
        indices = self.get_nearest_neighbors_title_and_body(phrase_embedding, n_results)
        results = [self.retrieve(i) for i in indices]
        return results

    def get_nearest_neighbors_title_and_body(self, phrase_embedding, n_results: int = 10) -> Sequence[int]:
        """Returns the (deduplicated) indices of the embeddings that are closest to
        the given phrase embedding."""
        distances, indices = self.embeddings.kneighbors(phrase_embedding, n_results)
        return indices

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

    def _embed(self, *args, **kwargs):
        if self.embedding_model is None:
            self._setup_embedding_model()
        return self.embedding_model.embed(*args, **kwargs)

    def _setup_embedding_model(self):
        self.embedding_model = EmbeddingModel(embedding_model)

    def _init_db(self):
        stmt = """
                CREATE TABLE websites(
                    claim_id INT NOT NULL,
                    url TEXT,
                    text TEXT,
                    query TEXT,
                    type TEXT,
                    text_embedding BLOB
                );
                """
        self.cur.execute(stmt)
        self.db.commit()

    def build_db(self, from_path: str):
        """Creates the SQLite database."""

        print("Building database...")

        if os.path.getsize(self.db_file_path) > 1e6:
            raise RuntimeError(f"{self.db_file_path} already exists! Not overwriting.")

        files = [f for f in iter_files(from_path)]
        self._init_db()

        self.read_queue = Queue()
        self.insert_queue = Queue()

        reader = Thread(target=self._read, args=(files,))
        processor = Thread(target=self._process_and_embed)
        inserter = Thread(target=self._insert)

        reader.start()
        processor.start()
        inserter.start()

        reader.join()
        processor.join()
        inserter.join()

        print(f"Done building DB.")
        print("Committing...")
        self.db.commit()

    def _read(self, files):
        for file in files:
            self.read_queue.put(get_contents(file))
        self.read_queue.put("done")

    def _process_and_embed(self):
        while (websites := self.read_queue.get()) != "done":
            for website in websites:
                claim_id = website["claim_id"]
                url = website["url"]
                text = " ".join(website["url2text"])
                query = website["query"]
                w_type = website["type"]
                text_embedding = self._embed(text, to_bytes=True) if text != '' else None
                data = (claim_id, url, text, query, w_type, text_embedding)
                self.insert_queue.put(data)
        self.insert_queue.put("done")

    def _insert(self):
        # Threads need to re-initialize any SQLite object
        db = sqlite3.connect(self.db_file_path, uri=True)
        cur = db.cursor()
        pbar = tqdm(total=500_000)  # up to 1K URLs per claim, 500 claims
        while (data := self.insert_queue.get()) != "done":
            cur.execute("INSERT INTO websites VALUES (?,?,?,?,?,?)", data)
            pbar.update()
        pbar.close()


def postprocess(result: Sequence) -> str:
    """Keeps only the body text of the articles and concatenates them to a single string."""
    converted = ""
    if len(result) > 0:
        for row in result:
            converted += f"{row[0]}\n{row[1]}\n"
    return converted


def df_embedding_to_np_embedding(df: pd.DataFrame, dimension: int) -> np.array:
    embeddings = np.zeros(shape=(len(df), dimension), dtype="float32")
    for i, embedding in enumerate(tqdm(df)):
        embeddings[i] = struct.unpack(f"{dimension}f", embedding)
    return embeddings


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    websites = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            websites.append(doc)
    return websites


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)
