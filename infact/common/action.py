from abc import ABC

class Action(ABC):
    """Executed by the actor. Performing an Action yields Evidence."""
    name: str
    description: str
    how_to: str
    format: str
    is_multimodal: bool = False

    def __str__(self):
        return f"{self.name}: {self.description}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class Search(Action):
    api: str
    query: str

    def __init__(self, query: str):
        assert (query[0] == '"' and query[-1] == '"')
        self.query = query[1:-1]

    def __str__(self):
        return f'{self.name}("{self.query}")'

    def __eq__(self, other):
        return isinstance(other, Search) and self.query == other.query and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.query))


class WebSearch(Search):
    name = "web_search"
    description = """Run an open web search on Google or DuckDuckGO to retrieve any related webpage."""
    how_to = """Do not use this with a previously used or similar query from previous web searches.
    If a previous web search did not yield any results, use a very different query."""
    format = """web_search("your web search query goes here")"""
    is_multimodal = False


class WikiDumpLookup(Search):
    name = "wiki_dump_lookup"
    description = """Look up something on the Wikipedia dump from 2017. Each article in the dump
    contains only the first few paragraphs of the article. In particular, the dump is incomplete
    and may miss much information. Use the dump to retrieve an article for a person, an entity, 
    an event etc."""
    how_to = """Do not repeat queries from previous `wiki_dump_lookup`s. If a previous
    `wiki_dump_lookup` did not yield enough results, use a very different query."""
    format = """wiki_dump_lookup("your wiki search query goes here")"""
    is_multimodal = False


class WikiLookup(Search):
    name = "wiki_lookup"
    description = """Look up something on Wikipedia to retrieve an article for a person, an 
    entity, an event etc."""
    how_to = """Do not use this with a previously used or similar query from previous wiki lookup. 
    If a previous wiki_lookup did not yield any results, use a very different query."""
    format = """wiki_lookup("your wiki search query goes here")"""
    is_multimodal = False


ACTION_REGISTRY = {
    WebSearch,
    WikiDumpLookup,
    WikiLookup,
}
