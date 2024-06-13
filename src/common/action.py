from abc import ABC


class Action(ABC):
    name: str
    description: str
    how_to: str
    format: str


class Search(Action, ABC):
    api: str
    query: str

    def __str__(self):
        return f"{self.name}: {self.query}"


class WebSearch(Search):
    name = "WEB_SEARCH"
    description = """Run an open web search on Google or DuckDuckGO to retrieve any related webpage."""
    how_to = """Do not use this with a previously used or similar query from previous WEB_SEARCHes.
    If a previous WEB_SEARCH did not yield any results, use a very different query."""
    format = "WEB_SEARCH: your web search query goes here"

    def __init__(self, query: str):
        self.query = query


class WikiDumpLookup(Search):
    name = "WIKI_DUMP_LOOKUP"
    description = """Look up something on the Wikipedia dump from 2017. Each article in the dump
    contains only the first few paragraphs of the article. In particular, the dump is incomplete
    and may miss much information. Use the dump to retrieve an article for a person, an entity, 
    an event etc."""
    how_to = """Do not repeat queries from previous `WIKI_DUMP_LOOKUP`s. If a previous
    `WIKI_DUMP_LOOKUP` did not yield enough results, use a very different query."""
    format = """WIKI_DUMP_LOOKUP: your wiki search query goes here"""

    def __init__(self, query: str):
        self.query = query


class WikiLookup(Search):
    name = "WIKI_LOOKUP"
    description = """Look up something on Wikipedia to retrieve an article for a person, an 
    entity, an event etc."""
    how_to = """Do not use this with a previously used or similar query from previous WIKI_LOOKUPs. 
    If a previous WIKI_LOOKUP did not yield any results, use a very different query."""
    format = """WIKI_LOOKUP: your wiki search query goes here"""

    def __init__(self, query: str):
        self.query = query


ACTION_REGISTRY = {
    WebSearch,
    WikiDumpLookup
}
