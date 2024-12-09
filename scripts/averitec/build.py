"""Builds the AVeriTeC Knowledge Base (KB) for all three variants (dev, train, test)."""

from infact.tools.search.knowledge_base import KnowledgeBase


if __name__ == '__main__':  # KB building uses multiprocessing
    print("Starting to build the AVeriTeC Knowledge Bases...")
    for variant in ["dev", "train", "test"]:
        kb = KnowledgeBase(variant)

        # Run sanity check
        kb.current_claim_id = 0
        result = kb.search("Apple", limit=10)
        assert len(result) == 10

        print(f"{variant} KB is ready for usage!")
