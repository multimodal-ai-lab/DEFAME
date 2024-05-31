import time
from safe.tools import query_serper
from safe.tools import duckduckgo

def run_queries(searcher, queries):
    start_time = time.time()
    for query in queries:
        searcher.run(query)
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

def compare_search_engines(serper_searcher, duckduck_searcher, queries):
    serper_time = run_queries(serper_searcher, queries)
    duckduck_time = run_queries(duckduck_searcher, queries)
    
    print(f"SerperAPI total time for 20 queries: {serper_time:.2f} seconds")
    print(f"DuckDuckGo total time for 20 queries: {duckduck_time:.2f} seconds")

# Example usage
serper_api_key = "0dcaae8b265bc5c5138e1587b397cbb11138891f"
num_searches = 1  # Number of searches to perform

# Create the searcher instances
serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
duckduck_searcher = duckduckgo.DuckDuckGo(max_results=num_searches)

# Define 20 sample queries
queries = [
    "What is the capital of France?",
    "Who won the Nobel Prize in Physics 2020?",
    "Top 10 programming languages in 2024",
    "How to make a perfect cup of coffee",
    "Latest news on climate change",
    "Python vs Java performance comparison",
    "Best practices for remote work",
    "History of the internet",
#    "How does blockchain technology work?",
#    "Benefits of a healthy diet",
#    "What are quantum computers?",
#    "Upcoming tech conferences in 2024",
#    "Impact of AI on jobs",
#    "How to start a garden at home",
#    "Most popular tourist destinations in 2024",
#    "How to improve mental health",
#    "Latest trends in renewable energy",
#    "How to learn a new language quickly",
#    "Best practices for cybersecurity",
#    "Advancements in medical technology"
]

# Compare the search engines
compare_search_engines(serper_searcher, duckduck_searcher, queries)