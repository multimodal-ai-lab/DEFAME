import requests
import json
from datetime import datetime
from config.globals import api_keys
from defame.evidence_retrieval.scraping.scraper import scraper

# Google Fact Check API endpoint
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Replace this with your actual API key
API_KEY = api_keys["google_api_key"] 

def fetch_recent_claims(query=None, max_age_days=None, page_token=None):
    """
    Fetch claims from Google Fact Check API based on a query or maxAgeDays.
    """
    params = {
        "key": API_KEY,
    }
    if query:
        params["query"] = query
    if max_age_days:
        params["maxAgeDays"] = max_age_days
    if page_token:
        params["pageToken"] = page_token

    response = requests.get(FACT_CHECK_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def scrape_claims_after_date(after_date, query=None):
    """
    Scrape claims made after a specific date.
    """
    days_since = (datetime.utcnow() - datetime.fromisoformat(after_date.rstrip("Z"))).days
    all_claims = []
    next_page_token = None

    while True:
        data = fetch_recent_claims(query=query, max_age_days=days_since, page_token=next_page_token)
        if not data or "claims" not in data:
            break

        all_claims.extend(data["claims"])
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return all_claims

def identify_claims_with_images(claims):
    """
    Identify claims with associated images in the claimReview.
    """
    for claim in claims:
        claim["hasImages"] = False  # Default assumption
        if "claimReview" in claim:
            for review in claim["claimReview"]:
                if review.get("url"):
                    scraped = scraper.scrape(review["url"])
                    if scraped.has_images():
                        claim["hasImages"] = True
                        claim["imageReferences"] = scraped.images
                        scraped.images
                    break
    return claims

def save_claims_to_json(claims, query, after_date, filename="claims.json"):
    """
    Save claims to a JSON file with query and after_date metadata.
    """
    output_data = {
        "query": query,
        "after_date": after_date,
        "claims": claims
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(claims)} claims to {filename}")

def main():
    # Define the query and after date
    after_date = "2023-01-01T00:00:00Z"
    query = "climate change"  # Replace with your search keyword or set to None

    print(f"Fetching claims after {after_date}...")
    claims = scrape_claims_after_date(after_date, query=query)

    # Identify claims with images
    claims_with_metadata = identify_claims_with_images(claims)

    # Save to JSON file
    save_claims_to_json(claims_with_metadata, query, after_date, filename="claims_with_images.json")

    print(f"Found {len(claims)} claims. Saved with metadata to 'claims_with_images.json'.")

if __name__ == "__main__":
    main()
