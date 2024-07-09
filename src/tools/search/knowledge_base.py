import json
import os.path
import pickle
import sqlite3
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Optional

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from config.globals import path_to_data
from tools.search.semantic_search_db import SemanticSearchDB, df_embedding_to_np_embedding
from tools.search.x import X, extract_username_from_url, extract_tweet_id_from_url, extract_search_query_from_url


class KnowledgeBase(SemanticSearchDB):
    """The AVeriTeC knowledge base consisting of 330,589 (deduplicated) sources."""
    name = 'averitec_kb'

    embedding_knn_path = path_to_data + "AVeriTeC/embedding_knn.pckl"

    def __init__(self, logger=None):
        super().__init__(logger=logger, db_file_path=path_to_data + "AVeriTeC/knowledge_base.db")
        self._load_embeddings()

    def retrieve(self, idx: int) -> (str, str, datetime):
        stmt = f"""
            SELECT url, text
            FROM sources
            WHERE ROWID = {idx + 1};
            """
        return *self._run_sql_query(stmt)[0], None

    def retrieve_by_url(self, url: str) -> Optional[str]:
        stmt = f"""
            SELECT text
            FROM sources
            WHERE url = ?;
            """
        result = self._run_sql_query(stmt, url)
        if result:
            return result[0]
        else:
            return None

    def _search_semantically(self, query_embedding, limit: int) -> list[int]:
        """Performs a vector search on the text embeddings."""
        distances, indices = self.embeddings.kneighbors(query_embedding, limit)
        return indices[0]

    def build_db(self, from_path: str):
        """Creates the SQLite database."""

        print("Building database...")

        if os.path.getsize(self.db_file_path) > 1e6:
            raise RuntimeError(f"{self.db_file_path} already exists! Not overwriting.")

        files = [f for f in iter_files(from_path)]
        if not self._is_initialized():
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

    def _init_db(self):
        stmt = """
                CREATE TABLE sources(
                    url TEXT UNIQUE,
                    text TEXT,
                    text_embedding BLOB
                );
                """
        self.cur.execute(stmt)
        self.db.commit()

    def _read(self, files):
        for file in files:
            self.read_queue.put(get_contents(file))
        self.read_queue.put("done")

    def _process_and_embed(self):
        known_urls = set()
        while (sources := self.read_queue.get()) != "done":
            # Only keep sources that are new (not in known_urls)
            filtered_sources = []
            for source in sources:
                url = source["url"]
                text = "\n".join(source["url2text"])
                if url not in known_urls:
                    filtered_sources.append((url, text))
                    known_urls.add(url)

            # Embed all the (non-empty) sources at once
            sources_df = pd.DataFrame(filtered_sources, columns=["url", "text"])
            is_not_empty_string = sources_df["text"].str.len() > 0
            texts_to_embed = sources_df.loc[is_not_empty_string, "text"]
            embeddings = self._embed_many(texts_to_embed.to_list(), to_bytes=True)
            sources_df["embedding"] = None  # initialize new, empty column
            sources_df.loc[is_not_empty_string, "embedding"] = embeddings

            # Send the processed data to the next worker
            self.insert_queue.put(sources_df)

        self.insert_queue.put("done")

    def _insert(self):
        # Re-initialize connections (threads need to do that for any SQLite object anew)
        db = sqlite3.connect(self.db_file_path, uri=True)
        cur = db.cursor()

        # As long as there comes data from the queue, insert it into the DB
        pbar = tqdm(total=331_000, smoothing=0.01)  # size already known
        while True:
            sources_df = self.insert_queue.get()
            if isinstance(sources_df, str) and sources_df == "done":
                break
            cur.executemany("INSERT OR IGNORE INTO sources VALUES (?,?,?)", sources_df.values.tolist())
            pbar.update(n=len(sources_df))
        pbar.close()

        print(f"Done building DB.")
        print("Committing...", end="")
        db.commit()
        print(" done!")

    def _deduplicate(self):
        """Removes all duplicate entries from the database."""
        print("Deduplicating (may take a while)...", end="")
        # Remove all duplicates
        stmt = """
        DELETE FROM sources
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM sources
            GROUP BY url
        )
        """
        self.cur.execute(stmt)
        print(" done!")

        # 2. Reset ROWIDs to become sequential again (i.e. remove gaps)
        print("Resetting ROWIDs (may also take a while)...", end="")
        stmt = """
        ALTER TABLE sources ADD COLUMN new_rowid INTEGER;
        
        WITH seq AS (
            SELECT ROWID, ROW_NUMBER() OVER (ORDER BY ROWID) AS new_id
            FROM sources  
        )
        UPDATE sources      
        SET new_rowid = (SELECT new_id FROM seq WHERE seq.ROWID = sources.ROWID);
        
        UPDATE sources
        SET ROWID = new_rowid;
        
        ALTER TABLE sources DROP COLUMN new_rowid;
        """
        self.cur.execute(stmt)
        self.db.commit()
        print(" done!")

        # 3. Remove existing kNN learner because IDs have changed
        if os.path.exists(self.embedding_knn_path):
            os.remove(self.embedding_knn_path)

    def _refill_twitter_instances(self):
        """By default, the X/Twitter instances are empty (due to X's terms of use). We need to
        reload the contents of the respective 1636 URLs and save them into the database."""
        # Get the IDs of the samples with Twitter URL
        stmt = """
        SELECT rowid, url FROM sources
        WHERE url LIKE "https://twitter.%";
        """
        rows = self._run_sql_query(stmt)

        # Initiate the Twitter API
        x = X()

        # For each instance, load the current content from the web, and add it to the DB
        for row_id, url in rows:
            if tweet_id := extract_tweet_id_from_url(url):
                tweet = x.get_tweet(tweet_id=tweet_id)
                text = tweet
            elif search_query := extract_search_query_from_url(url):
                results = x.search(search_query, limit=10)
                text = results
            elif username := extract_username_from_url(url):
                user_page = x.get_user_page(username=username)
                text = user_page
            else:
                raise ValueError(f"URL {url} does not contain any username, tweet ID or search query.")
            print(text)
            raise NotImplementedError

        self.db.commit()

    def _is_initialized(self) -> bool:
        stmt = """SELECT name FROM sqlite_master WHERE type='table' AND name='sources';"""
        result = self._run_sql_query(stmt)
        return len(result) > 0

    def is_empty(self) -> bool:
        if not self._is_initialized():
            return True
        stmt = """SELECT * FROM sources LIMIT 1;"""
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
        stmt = "SELECT ROWID, text_embedding FROM sources ORDER BY ROWID"
        embeddings = pd.read_sql_query(stmt, self.db)
        print("Reading embeddings...")
        embeddings = df_embedding_to_np_embedding(embeddings["text_embedding"], self.embedding_model.dimension)
        print("Setting up nearest neighbor learners...")
        self.embeddings = NearestNeighbors(n_neighbors=10).fit(embeddings)
        print("Saving kNN learner...")
        with open(self.embedding_knn_path, "wb") as f:
            pickle.dump(self.embeddings, f)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    searches = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            searches.append(doc)
    return searches


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
