from .common import SocialMediaPost, SocialMediaProfile
from .bluesky import Bluesky, bluesky

# Export the main classes for easy access
__all__ = [
    'SocialMediaPost',
    'SocialMediaProfile',
    'Bluesky',
    'bluesky',
]

# Lazy load scrapeMM integrations to avoid initialization issues
def get_x_scrapemm():
    """Get X scrapeMM integration (lazy loaded)."""
    try:
        from .x_scrapemm import get_x_scrapemm as _get_x
        return _get_x()
    except ImportError:
        return None

def get_reddit_scrapemm():
    """Get Reddit scrapeMM integration (lazy loaded)."""
    try:
        from .reddit_scrapemm import get_reddit_scrapemm as _get_reddit
        return _get_reddit()
    except ImportError:
        return None

# Initialize scrapeMM integrations for registration
# These need to be created so they get registered in RETRIEVAL_INTEGRATIONS
x_scrapemm = get_x_scrapemm()
reddit_scrapemm = get_reddit_scrapemm()

# Add to __all__ if they exist
if x_scrapemm:
    __all__.extend(['x_scrapemm', 'get_x_scrapemm'])
if reddit_scrapemm:
    __all__.extend(['reddit_scrapemm', 'get_reddit_scrapemm'])
