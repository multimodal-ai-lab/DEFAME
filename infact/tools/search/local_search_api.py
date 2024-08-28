from infact.tools.search.search_api import SearchAPI


class LocalSearchAPI(SearchAPI):
    is_local = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
