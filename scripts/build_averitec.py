"""Downloads and prepares the AVeriTeC knowledge base/store. Preparation
will take several hours on an A100 GPU due to embedding computation."""

from tools.search.knowledge_base import KnowledgeBase


if __name__ == '__main__':  # DB building uses multiprocessing
    kb = KnowledgeBase()
    kb.build_db()
