import json
import os.path
import pickle
import shutil
import zipfile
from datetime import datetime
from multiprocessing import Pool, Queue
from typing import Optional
from urllib.request import urlretrieve

import langdetect
import torch
from ezmm import MultimodalSequence
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from config.globals import data_root_dir, embedding_model
from defame.common import logger
from defame.common.embedding import EmbeddingModel
from defame.evidence_retrieval.integrations.search.local_search_platform import LocalSearchPlatform
from defame.utils.utils import my_hook
from .common import SearchResults, Query, WebSource

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


class KnowledgeBase(LocalSearchPlatform):
    """The AVeriTeC Knowledge Base (KB) used to retrieve evidence for fact-checks.
    It aims to simulate a web search engine."""

    name = "averitec_kb"
    description = """The AVeriTeC Knowledge Base (KB). It simulates a web search engine
        similar to Google. It accepts and returns only textual queries/sources."""

    embedding_knns: dict[int, NearestNeighbors]
    embedding_model: EmbeddingModel = None

    def __init__(self, variant,
                 device: str | torch.device = None,
                 max_search_results: int = None):
        super().__init__()
        self.variant = variant
        self.max_search_results = max_search_results

        # Setup paths and dirs
        self.kb_dir = data_root_dir / f"AVeriTeC/knowledge_base/{variant}/"
        os.makedirs(self.kb_dir, exist_ok=True)
        self.download_dir = self.kb_dir / "download"
        self.extracted_dir = self.kb_dir / "extracted"
        self.resources_dir = self.kb_dir / "resources"  # stores all .jsonl files extracted from the .zip in download
        self.embedding_knns_path = self.kb_dir / "embedding_knns.pckl"

        self.current_claim_id: Optional[
            int] = None  # defines the behavior of the KB by preselecting the claim-relevant sources

        # For speeding up data loading
        self.cached_resources = None
        self.cached_resources_claim_id = None

        self.device = device

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

            # Preprocess resource texts, keep only non-empty natural language resources
            resources_preprocessed = []
            for resource in resources:
                text = "\n".join(resource["url2text"])

                # Only keep samples with non-zero text length
                if not text:
                    continue

                if len(text) < 512:
                    try:
                        lang = langdetect.detect(text)
                    except langdetect.LangDetectException as e:
                        lang = None

                    if lang is None:
                        # Sample does not contain any meaningful natural language, therefore omit it
                        continue

                resource["url2text"] = text
                resources_preprocessed.append(resource)

            # Save into cache for efficiency
            self.cached_resources = resources_preprocessed
            self.cached_resources_claim_id = claim_id

        return self.cached_resources

    def _embed(self, *args, **kwargs):
        if self.embedding_model is None:
            self._setup_embedding_model()
        return self.embedding_model.embed(*args, **kwargs)

    def _embed_many(self, *args, **kwargs):
        if self.embedding_model is None:
            self._setup_embedding_model()
        return self.embedding_model.embed_many(*args, batch_size=32, **kwargs)

    def _setup_embedding_model(self):
        self.embedding_model = EmbeddingModel(embedding_model, device=self.device)

    def retrieve(self, idx: int) -> (str, str, datetime):
        resources = self._get_resources()
        resource = resources[idx]
        url, text, date = resource["url"], resource["url2text"], None
        return url, text, date

    def _indices_to_search_results(self, indices: list[int], query: Query) -> list[WebSource]:
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

    def _call_api(self, query: Query) -> Optional[SearchResults]:
        """Performs a vector search on the text embeddings of the resources of the currently active claim."""
        if self.current_claim_id is None:
            raise RuntimeError("No claim ID specified. You must set the current_claim_id to the "
                               "ID of the currently fact-checked claim.")

        knn = self.embedding_knns[self.current_claim_id]
        if knn is None:
            return None

        query_embedding = self._embed(query.text).reshape(1, -1)
        limit = query.limit or self.max_search_results
        limit = min(limit, knn.n_samples_fit_)  # account for very small resource sets
        try:
            distances, indices = knn.kneighbors(query_embedding, limit)
            sources = self._indices_to_search_results(indices[0], query)
            return SearchResults(sources=sources, query=query)
        except Exception as e:
            logger.warning(f"Resource retrieval from kNN failed: {e}")
            return None

    def _download(self):
        print("Downloading knowledge base...")
        os.makedirs(self.download_dir, exist_ok=True)
        urls = DOWNLOAD_URLS[self.variant]
        for i, url in enumerate(urls):
            target_path = self.download_dir / f"{i}.zip"
            urlretrieve(url, target_path, my_hook(tqdm()))

    def _extract(self):
        print("Extracting knowledge base...")
        if self.resources_dir.exists():
            shutil.rmtree(self.resources_dir)
        zip_files = os.listdir(self.download_dir)
        for zip_file in tqdm(zip_files):
            zip_path = self.download_dir / zip_file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extracted_dir)
        match self.variant:
            case "dev":
                os.rename(self.extracted_dir / f"output_dev", self.resources_dir)
            case "train":
                os.rename(self.extracted_dir / f"data_store/train", self.resources_dir)
            case "test":
                os.rename(self.extracted_dir, self.resources_dir)

    def _build(self):
        """Downloads, extracts and creates the SQLite database."""
        print(f"Building the {self.variant} knowledge base...")

        if (not self.download_dir.exists() or
                len(os.listdir(self.download_dir)) < len(DOWNLOAD_URLS[self.variant])):
            self._download()
        else:
            print("Found downloaded zip files.")

        if (not self.resources_dir.exists() or
                len(os.listdir(self.resources_dir)) < self.get_num_claims()):
            self._extract()
        else:
            print("Found extracted resource files.")

        if not self.embedding_knns_path.exists():
            n_workers = torch.cuda.device_count()

            print(f"Constructing kNNs for embeddings using {n_workers} workers...")

            # Initialize and run the KB building pipeline
            self.resource_queue = Queue()
            self.embedding_queue = Queue()
            devices_queue = Queue()

            with Pool(n_workers, embed, (self.resource_queue, self.embedding_queue, devices_queue)):
                for d in range(n_workers):
                    devices_queue.put(d)
                self._read()
                self._train_embedding_knn()

        else:
            self._restore()

        print(f"Successfully built the {self.variant} knowledge base!")

    def _read(self):
        print("Reading and preparing resource files...")
        for claim_id in tqdm(range(self.get_num_claims())):
            resources = self._get_resources(claim_id)
            self.resource_queue.put((claim_id, resources))

    def _train_embedding_knn(self):
        print("Fitting the k nearest neighbor learners...")

        # Re-initialize connections (threads need to do that for any SQLite object anew)
        embedding_knns = dict()

        # As long as there comes data from the queue, insert it into the DB
        for _ in tqdm(range(self.get_num_claims()), smoothing=0.01):
            out = self.embedding_queue.get()
            claim_id, embeddings = out
            if len(embeddings) > 0:
                embedding_knn = NearestNeighbors(n_neighbors=10).fit(embeddings)
            else:
                embedding_knn = None
            embedding_knns[claim_id] = embedding_knn

        with open(self.embedding_knns_path, "wb") as f:
            pickle.dump(embedding_knns, f)

        self.embedding_knns = embedding_knns

    def _restore(self):
        with open(self.embedding_knns_path, "rb") as f:
            self.embedding_knns = pickle.load(f)
        logger.log(f"Successfully restored knowledge base.")


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


def embed(in_queue: Queue, out_queue: Queue, devices_queue: Queue):
    device = devices_queue.get()
    em = EmbeddingModel(embedding_model, device=f"cuda:{device}")

    while True:
        claim_id, resources = in_queue.get()

        # Embed all the resources at once
        texts = [resource["url2text"] for resource in resources]
        embeddings = em.embed_many(texts, batch_size=32)

        # Send the processed data to the next worker
        out_queue.put((claim_id, embeddings))
