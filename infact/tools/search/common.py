from dataclasses import dataclass
from datetime import date

import numpy as np

from infact.common import MultimediaSnippet, Result, Action, Image
from infact.common.misc import WebSource


@dataclass
class SearchResult(Result):
    """A collection of web sources."""
    sources: list[WebSource]

    def __str__(self):
        return "\n\n".join(map(str, self.sources))


@dataclass
class ReverseSearchResult(SearchResult):
    """Ships with additional object detection information next to the list of web sources."""
    entities: dict[str, float]  # mapping between entity description and confidence score
    best_guess_labels: list[str]

    def __str__(self):
        string_repr = "Google Vision's outputs"
        if self.entities:
            string_repr += f"\nIdentified web entities (confidence in parenthesis):\n"
            string_repr += "\n".join(f"{description} ({confidence * 100:.0f} %)"
                                     for description, confidence in self.entities.items())
        if self.best_guess_labels:
            string_repr += (f"\nBest guess about the topic of "
                           f"the image is {', '.join(self.best_guess_labels)}")
        return string_repr + "\n\n" + "\n\n".join(map(str, self.sources))


class Search(Action):
    api: str
    query_string: str
    search_type: str
    start_date: date
    end_date: date

    def __init__(self, query: str, start_date: date = None, end_date: date = None):
        assert ((query[0] == '"' and query[-1] == '"') or (query[0] == '<' and query[-1] == '>'))
        self.query_string = query[1:-1]
        self.start_date = start_date
        self.end_date = end_date

    def __str__(self):
        return f'{self.name}("{self.query_string}")'

    def __eq__(self, other):
        return isinstance(other, Search) and self.query_string == other.query_string and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.query_string))


class WebSearch(Search):
    name = "web_search"
    search_type = 'search'
    description = """Run an open web search on Google or DuckDuckGO to retrieve any related webpage."""
    how_to = """Do not use this with a previously used or similar query from previous web searches.
    If a previous web search did not yield any results, use a very different query."""
    format = """web_search("your web search query goes here")"""
    is_multimodal = False


class ImageSearch(Search):
    name = "image_search"
    search_type = 'images'
    description = """Run an image search on Google to retrieve related images for a given query."""
    how_to = """This is a helpful tool for fact-checking images. Use this to retrieve images associated with a specific keyword or phrase. 
    Ensure that the query is clear and specific to improve search accuracy. 
    If no relevant images are found, refine the query or use more descriptive terms."""
    format = """image_search("your image search query goes here")"""
    is_multimodal = True


class WikiDumpLookup(Search):
    name = "wiki_dump_lookup"
    search_type = 'search'
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
    search_type = 'search'
    description = """Look up something on Wikipedia to retrieve an article for a person, an 
    entity, an event etc."""
    how_to = """Do not use this with a previously used or similar query from previous wiki lookup. 
    If a previous wiki_lookup did not yield any results, use a very different query."""
    format = """wiki_lookup("your wiki search query goes here")"""
    is_multimodal = False


class ReverseSearch(Search):
    name = "reverse_search"
    search_type = 'reverse'
    description = "Performs a reverse image search to find similar images on the web."
    how_to = "Provide an image and the model will perform a reverse search to find similar images."
    format = 'reverse_search(image)'
    is_multimodal = True

    def __init__(self, image_ref: str):
        super().__init__(query=image_ref)
        self.image: Image = MultimediaSnippet(image_ref).images[0]

    def __str__(self):
        return f'{self.name}()'

    def __eq__(self, other):
        return isinstance(other, ReverseSearch) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.image.tobytes()))
