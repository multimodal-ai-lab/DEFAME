from .common import SearchResults, Source, WebSource
from .duckduckgo import DuckDuckGo
from .google_search import Google
from .knowledge_base import KnowledgeBase
from .search_platform import SearchPlatform
from .wiki_dump import WikiDump
from .x_search import XSearch
from .reddit_search import RedditSearch

_PLATFORMS = [Google, DuckDuckGo, KnowledgeBase, WikiDump, XSearch, RedditSearch]

PLATFORMS = {platform.name: platform for platform in _PLATFORMS}
