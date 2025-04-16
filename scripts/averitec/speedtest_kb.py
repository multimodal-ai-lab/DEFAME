import time

from defame.evidence_retrieval.integrations.search.knowledge_base import KnowledgeBase

# Instantiate the KnowledgeBase
kb = KnowledgeBase(variant="dev")

# List of sample queries to test
queries = [
    "What is the capital of France?",
    "How many continents are there?",
    "Who is the president of the United States?",
    "What is the tallest mountain in the world?",
    "When was the Declaration of Independence signed?",
    "What is the population of China?",
    "How far is the moon from the Earth?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest ocean on Earth?",
    "When was the first moon landing?",
    "What is the speed of light?",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "What is the longest river in the world?",
    "What is the smallest country in the world?",
    "What is the most spoken language in the world?",
    "Who discovered penicillin?",
    "What is the formula for water?",
    "What is the atomic number of carbon?",
    "What is the largest desert in the world?"
]


# Function to measure search time for each query
def test_search_speed(tester, queries):
    for query in queries:
        start_time = time.time()  # Start the timer
        tester._search(query, limit=5)  # Perform the search
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Query: {query}\nTime taken: {elapsed_time:.4f} seconds\n")


# Run the speed test
test_search_speed(kb, queries)
