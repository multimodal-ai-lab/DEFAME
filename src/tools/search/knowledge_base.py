import json
import os.path
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from urllib.request import urlretrieve

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from config.globals import path_to_data, embedding_model
from src.common.embedding import EmbeddingModel
from src.common.results import SearchResult
from src.tools.search.local_search_api import LocalSearchAPI
from src.utils.utils import my_hook

DOWNLOAD_URLS = {
    "dev": [
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip"
    ],
    "train": [
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/train/train_0_999.zip",
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/train/train_1000_1999.zip",
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/train/train_2000_3067.zip",
    ],
    "test": [
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/test/test_0_499.zip",
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/test/test_500_999.zip",
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/test/test_1000_1499.zip",
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/test/test_1500_1999.zip",
        "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/test/test_2000_2214.zip",
    ]
}

N_CLAIMS = {
    "dev": 500,
    "train": 3068,
    "test": 2215,
}


class KnowledgeBase(LocalSearchAPI):
    """The AVeriTeC Knowledge Base (KB) used to retrieve evidence for fact-checks."""
    name = 'averitec_kb'
    embedding_knns: dict[int, NearestNeighbors]
    embedding_model: EmbeddingModel = None

    def __init__(self, variant: str = "dev", logger=None):
        super().__init__(logger=logger)
        self.variant = variant

        # Setup paths and dirs
        self.kb_dir = Path(path_to_data + f"AVeriTeC/knowledge_base/{variant}/")
        os.makedirs(self.kb_dir, exist_ok=True)
        self.download_dir = self.kb_dir / "download"
        self.resources_dir = self.kb_dir / "resources"  # stores all .jsonl files extracted from the .zip in download
        self.embedding_knns_path = self.kb_dir / "embedding_knns.pckl"

        self.current_claim_id = None  # defines the behavior of the KB by preselecting the claim-relevant sources

        # For speeding up data loading
        self.cached_resources = None
        self.cached_resources_claim_id = None

        self._load()

    def get_num_claims(self) -> int:
        """Returns the number of claims the knowledge base is holding resources for."""
        return N_CLAIMS[self.variant]

    def _load(self):
        if self.is_built():
            self._restore()
        else:
            self._build()

    def is_built(self) -> bool:
        """Returns true if the KB is built (KB files are downloaded and extracted and embedding kNNs are there)."""
        return (os.path.exists(self.resources_dir) and
                len(os.listdir(self.resources_dir)) == self.get_num_claims() and
                os.path.exists(self.embedding_knns_path))

    def _get_resources(self, claim_id: int = None) -> list[dict]:
        """Returns the list of resources for the currently active claim ID."""
        claim_id = self.current_claim_id if claim_id is None else claim_id

        if self.cached_resources_claim_id != claim_id:
            # Load resources from disk
            resource_file_path = self.resources_dir / f"{claim_id}.json"
            resources = get_contents(resource_file_path)

            # Preprocess resources
            for resource in resources:
                resource["url2text"] = "\n".join(resource["url2text"])

            # Save into cache for efficiency
            self.cached_resources = resources
            self.cached_resources_claim_id = claim_id

        return self.cached_resources

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

    def retrieve(self, idx: int) -> (str, str, datetime):
        resources = self._get_resources()
        resource = resources[idx]
        url, text, date = resource["url"], resource["url2text"], None
        return url, text, date

    def _indices_to_search_results(self, indices: list[int], query: str) -> list[SearchResult]:
        results = []
        for i, index in enumerate(indices):
            url, text, date = self.retrieve(index)
            result = SearchResult(
                source=url,
                text=text,
                query=query,
                rank=i,
                date=date
            )
            results.append(result)
        return results

    def _call_api(self, query: str, limit: int) -> list[SearchResult]:
        """Performs a vector search on the text embeddings of the resources of the currently active claim."""
        if self.current_claim_id is None:
            raise RuntimeError("No claim ID specified. You must set the current_claim_id to the "
                               "ID of the currently fact-checked claim.")
        knn = self.embedding_knns[self.current_claim_id]
        query_embedding = self._embed(query)
        distances, indices = knn.kneighbors(query_embedding, limit)
        return self._indices_to_search_results(indices[0], query)

    def _download(self):
        print("Downloading knowledge base...")
        os.makedirs(self.download_dir, exist_ok=True)
        urls = DOWNLOAD_URLS[self.variant]
        for i, url in enumerate(urls):
            target_path = self.download_dir / f"{i}.zip"
            urlretrieve(url, target_path, my_hook(tqdm()))

    def _extract(self):
        print("Extracting knowledge base...")
        # os.makedirs(self.resources_dir, exist_ok=True)
        zip_files = os.listdir(self.download_dir)
        for zip_file in tqdm(zip_files):
            zip_path = self.download_dir / zip_file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.kb_dir)
        os.rename(self.kb_dir / f"output_{self.variant}", self.resources_dir)

    def _build(self):
        """Downloads, extracts and creates the SQLite database."""
        if (not os.path.exists(self.download_dir) or
                len(os.listdir(self.download_dir)) < len(DOWNLOAD_URLS[self.variant])):
            self._download()
        else:
            print("Found downloaded zip files.")

        if (not os.path.exists(self.resources_dir) or
                len(os.listdir(self.resources_dir)) < self.get_num_claims()):
            self._extract()
        else:
            print("Found extracted resource files.")

        print("Constructing kNNs for embeddings...")

        # Initialize and run the KB building pipeline
        self.read_queue = Queue()
        self.train_queue = Queue()

        reader = Thread(target=self._read)
        processor = Thread(target=self._process_and_embed)
        trainer = Thread(target=self._train_embedding_knn)

        reader.start()
        processor.start()
        trainer.start()

        reader.join()
        processor.join()
        trainer.join()

    def _read(self):
        for claim_id in range(self.get_num_claims()):
            resources = self._get_resources(claim_id)
            self.read_queue.put(resources)
        self.read_queue.put("done")

    def _process_and_embed(self):
        while (resources := self.read_queue.get()) != "done":
            # Embed all the resources at once
            texts = [resource["url2text"] for resource in resources]
            embeddings = self._embed_many(texts)
            claim_id = int(resources[0]["claim_id"])

            # Send the processed data to the next worker
            self.train_queue.put((claim_id, embeddings))

        self.train_queue.put("done")

    def _train_embedding_knn(self):
        # Re-initialize connections (threads need to do that for any SQLite object anew)
        embedding_knns = dict()

        # As long as there comes data from the queue, insert it into the DB
        pbar = tqdm(total=self.get_num_claims(), smoothing=0.01)
        while (out := self.train_queue.get()) != "done":
            claim_id, embeddings = out
            embedding_knn = NearestNeighbors(n_neighbors=10).fit(embeddings)
            embedding_knns[claim_id] = embedding_knn
            pbar.update()
        pbar.close()

        with open(self.embedding_knns_path, "wb") as f:
            pickle.dump(embedding_knns, f)

        print(f"Successfully built the {self.variant} knowledge base!")

    def _restore(self):
        print("Restoring existing kNN learners... ", end="")
        with open(self.embedding_knns_path, "rb") as f:
            self.embedding_knns = pickle.load(f)
        print("done.")


def get_contents(file_path) -> list[dict]:
    """Parse the contents of a file. Each line is a JSON encoded document."""
    searches = []
    with open(file_path) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            searches.append(doc)
    return searches
