"""
Shared URL registry for passing URLs between evidence retrieval tools.
"""

# Global registry to store URLs found by different tools
_url_registry = {
    'reddit_urls': [],
    'x_urls': []
}

def add_reddit_urls(urls: list[str]):
    """Add Reddit URLs found during evidence retrieval."""
    global _url_registry
    _url_registry['reddit_urls'].extend(urls)
    # Remove duplicates
    _url_registry['reddit_urls'] = list(set(_url_registry['reddit_urls']))

def add_x_urls(urls: list[str]):
    """Add X/Twitter URLs found during evidence retrieval."""
    global _url_registry
    _url_registry['x_urls'].extend(urls)
    # Remove duplicates
    _url_registry['x_urls'] = list(set(_url_registry['x_urls']))

def get_reddit_urls() -> list[str]:
    """Get all Reddit URLs found during evidence retrieval."""
    return _url_registry['reddit_urls'].copy()

def get_x_urls() -> list[str]:
    """Get all X/Twitter URLs found during evidence retrieval."""
    return _url_registry['x_urls'].copy()

def clear_urls():
    """Clear all stored URLs (call at start of new fact-check)."""
    global _url_registry
    _url_registry = {
        'reddit_urls': [],
        'x_urls': []
    }
