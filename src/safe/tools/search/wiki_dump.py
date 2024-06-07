import json
import os.path
import pickle
import sqlite3
from multiprocessing import Pool as ProcessPool
from datetime import datetime

import numpy as np
import pandas as pd
import unicodedata
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from common.shared_config import path_to_data
from common.utils import replace
from safe.tools.search.semantic_search_db import SemanticSearchDB, df_embedding_to_np_embedding


class WikiDumpAPI(SemanticSearchDB):
    """Class for querying the local SQLite database from the FEVER challenge."""
    name = "wiki_dump"

    title_knn_path = path_to_data + "FEVER/title_knn.pckl"
    body_knn_path = path_to_data + "FEVER/body_knn.pckl"

    def __init__(self, **kwargs):
        super().__init__(db_file_path=path_to_data + "FEVER/wiki.db", **kwargs)
        self._load_embeddings()

    def _load_embeddings(self):
        if os.path.exists(self.title_knn_path) and os.path.exists(self.body_knn_path):
            self._restore_knn()
        elif not self.is_empty():
            self._build_knn()

    def _restore_knn(self):
        print("Restoring existing kNN learners... ", end="")
        self.title_embeddings = self._restore_knn_from(self.title_knn_path)
        self.body_embeddings = self._restore_knn_from(self.body_knn_path)
        print("done.")

    def _build_knn(self):
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

    def _search_semantically(self, query_embedding, limit: int = 10) -> list[int]:
        """Returns the (deduplicated) indices of the embeddings that are closest to
        the given phrase embedding for both, the titles and the bodies."""
        n_neighbors = limit // 2
        distances_title, indices_title = self.title_embeddings.kneighbors(query_embedding, n_neighbors)
        distances_body, indices_body = self.body_embeddings.kneighbors(query_embedding, n_neighbors)

        indices = np.asarray([indices_title, indices_body]).flatten()
        distances = np.asarray([distances_title, distances_body]).flatten()

        df = pd.DataFrame(data=dict(indices=indices, distances=distances))
        df.drop_duplicates(subset="indices", keep="first", inplace=True)
        df.sort_values(by="distances", inplace=True)

        return df["indices"].tolist()

    def _is_empty(self) -> bool:
        stmt = """SELECT * FROM articles LIMIT 1;"""
        rows = self._run_sql_query(stmt)
        return len(rows) == 0

    def retrieve(self, idx: int) -> (str, str, datetime):
        stmt = f"""
            SELECT title, body
            FROM articles
            WHERE ROWID = {idx + 1};
            """
        title, body = self._run_sql_query(stmt)[0]
        url = title
        text = f"{title}\n{body}"
        return url, text, None

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
                title_embeddings = self.embedding_model.embed_many(titles, to_bytes=True, batch_size=len(pairs) // 8)
                body_embeddings = self.embedding_model.embed_many(bodies, to_bytes=True, batch_size=len(pairs) // 8)
                rows = zip(titles, bodies, title_embeddings, body_embeddings)
                cur.executemany("INSERT INTO articles VALUES (?,?,?,?)", rows)
                pbar.update()

        print(f"Done reading {count} articles.")
        print("Committing...")
        db.commit()
        db.close()


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


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    articles = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the articles list
            articles.append((normalize(doc['id']), doc['text']))
    return articles


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
