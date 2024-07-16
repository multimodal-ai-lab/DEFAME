from src.tools.search.search_api import SearchAPI


class RemoteSearchAPI(SearchAPI):
    is_local = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
