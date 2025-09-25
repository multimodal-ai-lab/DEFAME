from .base import SocialMedia  # <-- CHANGE THIS LINE
from .bluesky import Bluesky, bluesky
from .reddit import Reddit, reddit
from .x import X, x

# Export the main classes for easy access
__all__ = [
    'SocialMedia',
    'Bluesky',
    'Reddit', 
    'X',
    'bluesky',  # Export singleton instances so they get registered
    'reddit',   # Export singleton instances so they get registered  
    'x',        # Export singleton instances so they get registered
]