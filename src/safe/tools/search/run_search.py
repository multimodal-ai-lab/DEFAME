from safe.tools.search.knowledge_base import KnowledgeBase
from safe.tools.search.query_serper import SerperAPI
from safe.tools.search.wiki_dump import WikiDumpAPI
from safe.tools.search.duckduckgo import DuckDuckGo


search_engine = WikiDumpAPI()

while True:
    query = input(f"Enter a query for {search_engine.name}: ")
    results = search_engine.search(query, 10)
    for result in results:
        print(str(result) + "\n")
