import os.path
import pickle
import re
import sqlite3
import struct
import json
from typing import Sequence
from multiprocessing import Pool as ProcessPool

import numpy as np
import unicodedata
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from common.shared_config import path_to_data, embedding_model
from common.embedding import EmbeddingModel


class WikiDumpAPI:
    """Class for querying the local SQLite database from the FEVER challenge."""
    db_file_path = path_to_data + "FEVER/wiki.db"
    title_knn_path = path_to_data + "FEVER/title_knn.pckl"
    body_knn_path = path_to_data + "FEVER/body_knn.pckl"

    def __init__(self):
        self.embedding_model = EmbeddingModel(embedding_model)
        if not os.path.exists(self.db_file_path):
            print("Warning: No FEVER database found. Continuing without database.")
            return
        self.db = sqlite3.connect(self.db_file_path, uri=True)
        self.cur = self.db.cursor()
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
        title_embeddings = df_embedding_to_np_embedding(embeddings["title_embedding"],
                                                        self.embedding_model.dimension)
        print("Reading body embeddings...")
        self.embedding = df_embedding_to_np_embedding(embeddings["body_embedding"], self.embedding_model.dimension)
        body_embeddings = self.embedding
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
        phrase_embedding = self.embedding_model.embed(phrase).reshape(1, -1)
        indices = self.get_nearest_neighbors_title_and_body(phrase_embedding, n_results)
        results = [self.retrieve(i) for i in indices]
        return results

    def get_nearest_neighbors_title_and_body(self, phrase_embedding, n_results: int = 10) -> Sequence[int]:
        """Returns the (deduplicated) indices of the embeddings that are closest to
        the given phrase embedding for both, the titles and the bodies."""
        n_neighbors = n_results // 2
        distances_title, indices_title = self.title_embeddings.kneighbors(phrase_embedding, n_neighbors)
        distances_body, indices_body = self.body_embeddings.kneighbors(phrase_embedding, n_neighbors)

        indices = np.asarray([indices_title, indices_body]).flatten()
        distances = np.asarray([distances_title, distances_body]).flatten()

        df = pd.DataFrame(data=dict(indices=indices, distances=distances))
        df.drop_duplicates(subset="indices", keep="first", inplace=True)
        df.sort_values(by="distances", inplace=True)

        return df["indices"].tolist()

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

    def build_db(self, from_path: str, num_workers: int = 4):
        """Creates the SQLite database."""

        print("Fetching resource files...")

        files = [f for f in iter_files(from_path)]

        print("Building database...")

        if os.path.isfile(self.db_file_path):
            raise RuntimeError(f"{self.db_file_path} already exists! Not overwriting.")

        os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
        db = sqlite3.connect(self.db_file_path)
        cur = db.cursor()
        stmt = """
        CREATE TABLE articles(
            title TEXT PRIMARY KEY,
            body TEXT,
            title_embedding BLOB,
            body_embedding BLOB
        );
        """
        cur.execute(stmt)

        workers = ProcessPool(num_workers)
        count = 0
        with tqdm(total=len(files)) as pbar:
            for pairs in tqdm(workers.imap_unordered(get_contents, files)):
                count += len(pairs)
                titles = [process_title(pair[0]) for pair in pairs]
                bodies = [process_body(pair[1]) for pair in pairs]
                title_embeddings = embedding_model.embed_many(titles, to_bytes=True, batch_size=len(pairs) // 8)
                body_embeddings = embedding_model.embed_many(bodies, to_bytes=True, batch_size=len(pairs) // 8)
                rows = zip(titles, bodies, title_embeddings, body_embeddings)
                cur.executemany("INSERT INTO articles VALUES (?,?,?,?)", rows)
                pbar.update()

        print(f"Done reading {count} articles.")
        print("Committing...")
        db.commit()
        db.close()


def postprocess(result: Sequence) -> str:
    """Keeps only the body text of the articles and concatenates them to a single string."""
    converted = ""
    if len(result) > 0:
        for row in result:
            converted += f"{process_title(row[0])}\n{process_body(row[1])}\n"
    return converted


def process_title(title: str) -> str:  # Do not change! It will change the embeddings
    title = title.replace("_", " ")
    return process_body(title)


def process_body(body: str) -> str:  # Do not change! It will change the embeddings
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


def df_embedding_to_np_embedding(df: pd.DataFrame, dimension: int) -> np.array:
    embeddings = np.zeros(shape=(len(df), dimension), dtype="float32")
    for i, embedding in enumerate(tqdm(df)):
        embeddings[i] = struct.unpack(f"{dimension}f", embedding)
    return embeddings


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((normalize(doc['id']), doc['text']))
    return documents


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
