from safe.tools.wiki_dump import WikiDumpAPI

wiki = WikiDumpAPI()

while True:
    query = input("Enter a Wiki query: ")
    print(wiki.search(query, 10))
