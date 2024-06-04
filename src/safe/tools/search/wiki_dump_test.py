from wiki_dump import WikiDumpAPI

wiki_dump_api = WikiDumpAPI()
test_queries = ["Germany", "Kennedy was shot dead", "Eurovision Song Contest winner 2015", "The earth is flat!"]

for query in test_queries:
    print(f"Query: '{query}'")
    print(wiki_dump_api.search_semantically(query))
