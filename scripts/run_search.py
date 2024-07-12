from src.tools.search.wiki_dump import WikiDumpAPI

search_engine = WikiDumpAPI()

while True:
    query = input(f"Enter a query for {search_engine.name}: ")
    results = search_engine.search(query, 10)
    for result in results:
        print(str(result) + "\n")
