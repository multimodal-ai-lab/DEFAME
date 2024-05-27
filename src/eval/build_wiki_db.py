# Adapted and modified from https://github.com/jongminyoon/fever
# which is adapted from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

"""A script to read in and store documents in a sqlite database."""

import argparse
import json
import logging
import os
import sqlite3
from multiprocessing import Pool as ProcessPool

import unicodedata
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Build db")


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


def store_contents(data_path: str, save_dir: str, num_workers: int = 4):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_dir: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(text):
        embedded = model.encode(text, show_progress_bar=False)
        embedded_bytes = [e.tobytes() for e in embedded]
        return embedded_bytes

    logger.info('Reading into database...')

    files = [f for f in iter_files(data_path)]

    logger.info('Building db...')

    db_path = os.path.join(save_dir, 'wiki.db')

    if os.path.isfile(db_path):
        raise RuntimeError('%s already exists! Not overwriting.' % db_path)

    db = sqlite3.connect(db_path)
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
            titles = [pair[0] for pair in pairs]
            bodies = [pair[1] for pair in pairs]
            title_embeddings = embed(titles)
            body_embeddings = embed(bodies)
            rows = zip(titles, bodies, title_embeddings, body_embeddings)
            cur.executemany("INSERT INTO articles VALUES (?,?,?,?)", rows)
            pbar.update()

    logger.info('Done reading %d docs.' % count)
    logger.info('Committing...')
    db.commit()
    db.close()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path/to/data')
    parser.add_argument('save_path', type=str, help='path/to/saved')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    save_dir = args.save_path

    if not os.path.exists(save_dir):
        logger.info("Save directory doesn't exist. Making {0}".format(save_dir))
        os.makedirs(save_dir)

    store_contents(args.data_path, args.save_path, args.num_workers)
