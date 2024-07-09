from tools.search.knowledge_base import KnowledgeBase
from config.globals import path_to_data


if __name__ == '__main__':  # DB building uses multiprocessing
    kb = KnowledgeBase()
    kb.build_db(path_to_data + "AVeriTeC/knowledge_dev/")
