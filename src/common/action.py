from abc import ABC
from PIL import Image
import numpy as np


class Action(ABC):
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
        return f"{self.name}: \"{self.query}\""

    def __eq__(self, other):
        return isinstance(other, Search) and self.query == other.query and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.query))


class WebSearch(Search):
    name = "web_search"
    description = """Run an open web search on Google or DuckDuckGO to retrieve any related webpage."""
    how_to = """Do not use this with a previously used or similar query from previous WEB_SEARCHes.
    If a previous WEB_SEARCH did not yield any results, use a very different query."""
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
    how_to = """Do not use this with a previously used or similar query from previous WIKI_LOOKUPs. 
    If a previous wiki_lookup did not yield any results, use a very different query."""
    format = """wiki_lookup("your wiki search query goes here")"""
    is_multimodal = False


class ObjectRecognition(Action):
    name = "object_recognition"
    description = "Identifies objects within an image."
    how_to = "Provide an image and the model will recognize objects in it."
    format = 'object_recognition(image)'
    is_multimodal = True

    def __init__(self, image: Image.Image):
        self.image = image

    def __str__(self):
        return f'{self.name}()'

    def __eq__(self, other):
        return isinstance(other, ObjectRecognition) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.tobytes()))


class ReverseSearch(Action):
    name = "reverse_search"
    description = "Performs a reverse image search to find similar images on the web."
    how_to = "Provide an image and the model will perform a reverse search to find similar images."
    format = 'reverse_search(image)'
    is_multimodal = True

    def __init__(self, image: Image.Image):
        self.image = image

    def __str__(self):
        return f'{self.name}()'

    def __eq__(self, other):
        return isinstance(other, ReverseSearch) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.tobytes()))

class GeoLocation(Action):
    name = "geo_location"
    description = "Performs geolocation to determine the country where an image was taken."
    how_to = f"Provide an image and the model will determine the most likely countries where it was taken."
    format = 'geo_location(image)'
    is_multimodal = True

    def __init__(self, image: Image.Image, top_k: int = 10):
        self.image = image
        self.top_k = top_k

    def __str__(self):
        return f'{self.name}()'

    def __eq__(self, other):
        return isinstance(other, GeoLocation) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.tobytes()))

class FaceRecognition(Action):
    name = "face_recognition"
    description = "Identifies and recognizes faces within an image."
    how_to = "Provide an image and the model will recognize faces in it."
    format = 'face_recognition(image_patch)'
    is_multimodal = True

    def __init__(self, image: Image.Image):
        self.image = image

    def __str__(self):
        return f'{self.name}()'

    def __eq__(self, other):
        return isinstance(other, FaceRecognition) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.tobytes()))
    
class OCR(Action):
    name = "ocr"
    description = "Performs Optical Character Recognition to extract text from an image."
    how_to = "Provide an image and the model will extract text from it."
    format = 'ocr(image)'
    is_multimodal = True

    def __init__(self, image: Image.Image):
        self.image = image

    def __str__(self):
        return f'{self.name}()'

    def __eq__(self, other):
        return isinstance(other, OCR) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.tobytes()))

class SourceCredibilityCheck(Action):
    name = "source_credibility_check"
    description = "Evaluates the credibility of a given source."
    how_to = "Provide a source URL or name and the model will assess its credibility."
    format = 'source_credibility_check("source_name_or_url")'
    is_multimodal = False

    def __init__(self, source: str):
        self.source = source

    def __str__(self):
        return f'{self.name}("{self.source}")'
    
    def __eq__(self, other):
        return isinstance(other, SourceCredibilityCheck) and self.source == other.source

    def __hash__(self):
        return hash((self.name, self.source))

    
ACTION_REGISTRY = {
    WebSearch,
    WikiDumpLookup,
    ObjectRecognition,
    WikiLookup,
    ReverseSearch,
    GeoLocation,
    FaceRecognition,
    SourceCredibilityCheck,
    OCR,
}
