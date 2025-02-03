from defame.tools.search.remote_search_api import _firecrawl_is_running, scrape_firecrawl

def test_firecrawl():
    assert _firecrawl_is_running()
    print(scrape_firecrawl("https://edition.cnn.com/2025/02/03/politics/usaid-washington-workers/index.html"))
