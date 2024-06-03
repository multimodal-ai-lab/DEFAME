from safe.tools.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

while True:
    query = input("Enter a query for the AVeriTeC knowledge base: ")
    print(kb.search(query, 10))
