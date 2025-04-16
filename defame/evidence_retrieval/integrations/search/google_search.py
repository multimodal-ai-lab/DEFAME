from typing import Optional

from defame.evidence_retrieval.integrations.search.serper import serper_api
from defame.evidence_retrieval.integrations.search.google_vision import google_vision_api
from defame.evidence_retrieval.integrations.search.common import Query, SearchResults
from defame.evidence_retrieval.integrations.search.remote_search_platform import RemoteSearchPlatform


class Google(RemoteSearchPlatform):
    """Integrates Serper API and Google Vision API into one search platform. Chooses
    the proper API automatically depending on the query."""
    name = "google"
    description = """The Google Search Engine. Use it to retrieve webpages,
        images, perform RIS and more."""

    def __init__(self, enable_ris: bool = True, **kwargs):
        """
        @param enable_ris: If True, enables Reverse Image Search (RIS).
        """
        self.enable_ris = enable_ris
        super().__init__(**kwargs)

    def _call_api(self, query: Query) -> Optional[SearchResults]:
        if self.enable_ris and query.has_image():
            return google_vision_api.search(query)
        else:
            return serper_api.search(query)
