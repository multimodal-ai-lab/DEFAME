from .common import SearchResults
from .duckduckgo import DuckDuckGo
from .google_vision_api import GoogleVisionAPI
from .knowledge_base import KnowledgeBase
from .search_api import SearchAPI
from .serper import SerperAPI
from .wiki_dump import WikiDump

SEARCH_ENGINES = {
    "duckduckgo": DuckDuckGo,
    "google_vision": GoogleVisionAPI,
    "averitec_kb": KnowledgeBase,
    "google": SerperAPI,
    "wiki_dump": WikiDump,
}
