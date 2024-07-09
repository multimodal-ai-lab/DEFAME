import json
import os.path
import pickle
import sqlite3
from queue import Queue
from threading import Thread

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from common.shared_config import path_to_data
from safe.tools.search.semantic_search_db import SemanticSearchDB, df_embedding_to_np_embedding


class KnowledgeBase(SemanticSearchDB):
    """The AVeriTeC knowledge base consisting of a selection of webpages."""
    name = 'averitec_kb'

    embedding_knn_path = path_to_data + "AVeriTeC/embedding_knn.pckl"

    def __init__(self, **kwargs):
        super().__init__(db_file_path=path_to_data + "AVeriTeC/knowledge_base.db", **kwargs)
        self._load_embeddings()

    def _is_empty(self) -> bool:
        stmt = """SELECT * FROM websites LIMIT 1;"""
        rows = self._run_sql_query(stmt)
        return len(rows) == 0

    def _load_embeddings(self):
        if os.path.exists(self.embedding_knn_path):
            self._restore_knn()
        elif not self.is_empty():
            self._build_knn()

    def _restore_knn(self):
        print("Restoring existing kNN learner... ", end="")
        self.embeddings = self._restore_knn_from(self.embedding_knn_path)
        print("done.")

    def _build_knn(self):
        self._setup_embedding_model()
        stmt = "SELECT ROWID, text_embedding FROM websites ORDER BY ROWID"
        embeddings = pd.read_sql_query(stmt, self.db)
        print("Reading embeddings...")
        embeddings = df_embedding_to_np_embedding(embeddings["text_embedding"], self.embedding_model.dimension)
        print("Setting up nearest neighbor learners...")
        self.embeddings = NearestNeighbors(n_neighbors=10).fit(embeddings)
        print("Saving kNN learner...")
        with open(self.embedding_knn_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def _search_semantically(self, query_embedding, limit: int) -> list[int]:
        """Performs a vector search on the text embeddings."""
        distances, indices = self.embeddings.kneighbors(query_embedding, limit)
        return indices[0]

    def retrieve(self, idx: int) -> (str, str):
        stmt = f"""
            SELECT url, text
            FROM websites
            WHERE ROWID = {idx + 1};
            """
        return self._run_sql_query(stmt)[0]

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
        pbar = tqdm(total=506_640, smoothing=0.1)  # size already known
        while (data := self.insert_queue.get()) != "done":
            cur.execute("INSERT INTO websites VALUES (?,?,?,?,?,?)", data)
            pbar.update()
        pbar.close()
        print(f"Done building DB.")
        print("Committing...", end="")
        db.commit()
        print(" done!")


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
