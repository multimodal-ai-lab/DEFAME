from defame.evidence_retrieval.integrations.search_engines.search_api import SearchAPI


class LocalSearchAPI(SearchAPI):
    is_local = True
