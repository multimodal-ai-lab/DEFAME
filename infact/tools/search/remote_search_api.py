from infact.tools.search.search_api import SearchAPI


class RemoteSearchAPI(SearchAPI):
    is_local = False
