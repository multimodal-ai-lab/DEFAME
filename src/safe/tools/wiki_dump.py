import os.path
import sqlite3
import re
from typing import Sequence
from common.shared_config import path_to_data


class WikiDumpAPI:
    """Class for querying the local SQLite database from the FEVER challenge."""
    def __init__(self):
        database_path = path_to_data + "FEVER/wiki-database/fever0.db"
        if not os.path.exists(database_path):
            print("Warning: No FEVER database found. Continuing without database.")
            return
        db = sqlite3.connect(path_to_data + "FEVER/wiki-database/fever0.db", uri=True)
        self.cur = db.cursor()

    def search(self, phrase: str) -> str:
        print(f"Searching Wiki dump with query: {phrase}")
        result = self._search_exact_title(phrase)
        if len(result) == 0:
            result = self._search_closest_title(phrase)
        if len(result) == 0:
            result = self._search_full_text(phrase)
        return postprocess(result)

    def _search_exact_title(self, phrase: str) -> Sequence:
        phrase = '"' + phrase + '"'

        query = """
            SELECT *
            FROM documents
            WHERE title MATCH ? AND length(title) = ?
            LIMIT 1;
            """

        return self._run_sql_query(query, phrase, len(phrase))

    def _search_closest_title(self, phrase: str) -> Sequence:
        phrase = preprocess(phrase)

        query = """
            SELECT *
            FROM documents
            WHERE title MATCH ?
            ORDER BY length(title), rank
            LIMIT 10;
            """

        return self._run_sql_query(query, phrase)

    def _search_full_text(self, phrase: str) -> Sequence:
        phrase = preprocess(phrase)

        query = """
            SELECT *
            FROM documents
            WHERE text MATCH ?
            ORDER BY rank
            LIMIT 10;
            """

        return self._run_sql_query(query, phrase)

    def _run_sql_query(self, query, *args):
        self.cur.execute(query, args)
        rows = self.cur.fetchall()
        return rows


def preprocess(phrase: str) -> str:
    """Preprocess phrase to remove special characters"""
    phrase = re.sub(r'[^a-zA-Z0-9]', " ", phrase)
    return phrase


def postprocess(result: Sequence) -> str:
    converted = ""
    if len(result) > 0:
        for row in result:
            converted += f" {row[1]}"
    return converted
