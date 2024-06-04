import time
from duckduckgo_search import DDGS
from typing import Any

def test_duckduckgo_search(initial_query: str, max_retries: int = 20, sleep_interval: int = 0) -> Any:
    current_query = initial_query
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nAttempt {attempt}: Searching for '{current_query}'")
            response = DDGS().text(current_query, max_results=3)
            print("Response received:", response)
        except Exception as e:
            print(f"Attempt {attempt} failed with exception: {e}\n")
            # Modify the request text slightly to avoid repeated failures with the same query
            current_query = current_query + "?"
            # Sleep for the specified interval before retrying
            if sleep_interval > 0:
                time.sleep(sleep_interval)
    print("Done.")
    return None

# Example usage
initial_query = "sean connery apple commercial refusal"
test_duckduckgo_search(initial_query, max_retries=50, sleep_interval=1)
