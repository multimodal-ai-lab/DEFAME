from src.tools.search.knowledge_base import KnowledgeBase


if __name__ == '__main__':  # KB building uses multiprocessing
    kb = KnowledgeBase("train")
