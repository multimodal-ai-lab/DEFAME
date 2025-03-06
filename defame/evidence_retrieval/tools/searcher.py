import re
from datetime import date
from typing import Any, Optional

import numpy as np
from jinja2.exceptions import TemplateSyntaxError
from openai import APIError

from config.globals import api_keys
from defame.common import MultimediaSnippet, Report, Prompt, logger, Action, Image
from defame.evidence_retrieval.integrations.search_engines.common import Query, WebSource, SearchMode
from defame.evidence_retrieval.tools.tool import Tool
from defame.evidence_retrieval.integrations.search_engines import (SearchResults, SearchAPI, SEARCH_ENGINES,
                                                                   GoogleVisionAPI)
from defame.prompts.prompts import SummarizeResultPrompt
from defame.utils.console import gray, orange


class Search(Action):
    api: str
    query_string: str
    search_mode: SearchMode
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
    search_mode = SearchMode.SEARCH
    description = """Run an open web search on Google or DuckDuckGO to retrieve any related webpage."""
    how_to = """Do not use this with a previously used or similar query from previous web searches.
    If a previous web search did not yield any results, use a very different query."""
    format = """web_search("your web search query goes here")"""
    is_multimodal = False
    is_limited = False


class ImageSearch(Search):
    name = "image_search"
    search_mode = SearchMode.IMAGES
    description = """Run an image search on Google to retrieve related images for a given query."""
    how_to = """This is a helpful tool for fact-checking images. Use this to retrieve images associated with a specific keyword or phrase. 
    Ensure that the query is clear and specific to improve search accuracy. 
    If no relevant images are found, refine the query or use more descriptive terms."""
    format = """image_search("your image search query goes here")"""
    is_multimodal = True
    is_limited = False


class WikiDumpLookup(Search):
    name = "wiki_dump_lookup"
    search_mode = SearchMode.SEARCH
    description = """Look up something on the Wikipedia dump from 2017. Each article in the dump
    contains only the first few paragraphs of the article. In particular, the dump is incomplete
    and may miss much information. Use the dump to retrieve an article for a person, an entity, 
    an event etc."""
    how_to = """Do not repeat queries from previous `wiki_dump_lookup`s. If a previous
    `wiki_dump_lookup` did not yield enough results, use a very different query."""
    format = """wiki_dump_lookup("your wiki search query goes here")"""
    is_multimodal = False
    is_limited = False


class WikiLookup(Search):
    name = "wiki_lookup"
    search_mode = SearchMode.SEARCH
    description = """Look up something on Wikipedia to retrieve an article for a person, an 
    entity, an event etc."""
    how_to = """Do not use this with a previously used or similar query from previous wiki lookup. 
    If a previous wiki_lookup did not yield any results, use a very different query."""
    format = """wiki_lookup("your wiki search query goes here")"""
    is_multimodal = False
    is_limited = False


class ReverseSearch(Search):
    name = "reverse_search"
    search_mode = SearchMode.REVERSE
    description = "Performs a reverse image search to find similar images on the web."
    how_to = "Provide an image and the model will perform a reverse search to find similar images."
    format = "reverse_search(<image:k>), where `k` is the image's ID"
    is_multimodal = True
    is_limited = True

    def __init__(self, image_ref: str):
        super().__init__(query=image_ref)
        self.image: Image = MultimediaSnippet(image_ref).images[0]

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, ReverseSearch) and np.array_equal(np.array(self.image), np.array(other.image))

    def __hash__(self):
        return hash((self.name, self.image.image.tobytes()))


class Searcher(Tool):
    """Searches the specified resource (Google, Wikipedia, ...) for evidence. Takes
    a list of specified search engines. The list defines the precedence of the search
    engines meaning that if search_engine[0] did not yield any results, search_engine[1]
    will be tried next."""
    # TODO: Rank or annotate the websites according to their credibility, like MUSE
    name = "searcher"
    summarize = True
    search_apis: dict[str, SearchAPI]
    stats: dict[str, int]

    def __init__(self,
                 search_engine_config: dict[str, dict] = None,
                 summarize: bool = True,
                 max_searches: int = 5,
                 limit_per_search: int = 5,
                 max_result_len: int = None,  # chars
                 do_debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        if search_engine_config is None:
            if api_keys["serper_api_key"]:
                search_engine_config = {"google": {}}
            else:
                logger.log("No Serper API key provided. Falling back to DuckDuckGo.")
                search_engine_config = {"duckduckgo": {}}

        # Add device to knowledge base kwargs
        if "averitec_kb" in search_engine_config:
            search_engine_config["averitec_kb"].update(dict(device=self.device))

        # Setup search APIs
        self.search_apis = {}
        for se, kwargs in search_engine_config.items():
            api_class = SEARCH_ENGINES[se]
            if kwargs is None:
                kwargs = {}
            api = api_class(max_search_results=limit_per_search, **kwargs)
            self.search_apis[se] = api

        # Register available tools
        actions = []
        available_apis = self.search_apis.keys()
        if "wiki_dump" in available_apis:
            actions.append(WikiDumpLookup)
        if "google" in available_apis:
            actions += [WebSearch, ImageSearch]
        if "duckduckgo" in available_apis or "averitec_kb" in available_apis:
            actions.append(WebSearch)
        if "google_vision" in available_apis:
            actions.append(ReverseSearch)
        self.actions = actions

        self.summarize = summarize
        self.max_searches = max_searches
        self.limit_per_search = limit_per_search
        self.max_result_len = max_result_len  # chars
        self.restrict_results_until_date = None

        self.debug = do_debug

        # Cut the result text, maintaining a little buffer for the summary prompt
        self.known_web_sources = set()

        self.reset()

    def _perform(self, action: Search) -> SearchResults:
        # Pick the strictest specified end date
        end_date = self.restrict_results_until_date
        if action.end_date is not None:
            if end_date is None or action.end_date < end_date:
                end_date = action.end_date

        image = action.image if isinstance(action, ReverseSearch) else None

        query = Query(
            text=action.query_string,
            image=image,
            search_mode=action.search_mode,
            limit=self.limit_per_search,
            start_date=action.start_date,
            end_date=end_date
        )

        return self.search(query)

    def search(self, query: Query) -> SearchResults:
        """Searches for evidence using the search APIs according to their precedence."""

        # TODO: Convert into a dict lookup
        for search_engine in list(self.search_apis.values()):
            if query.search_mode == SearchMode.REVERSE and isinstance(search_engine, GoogleVisionAPI):
                search_result = search_engine.search(query)
            elif (query.search_mode == SearchMode.SEARCH or query.search_mode == SearchMode.IMAGES and
                  not isinstance(search_engine, GoogleVisionAPI)):
                search_result = search_engine.search(query)
            else:
                continue

            # Remove known websites from search result
            search_result.sources = self._remove_known_web_sources(search_result.sources)

            # Track search engine call
            self.stats[search_engine.name] += 1

            # Log search result info
            logger.log(f"Got {len(search_result.sources)} new web source(s):")
            for i, web_source in enumerate(search_result.sources):
                result_summary = f"\t{i + 1}."
                if web_source.date is not None:
                    result_summary += f" {web_source.date.strftime('%B %d, %Y')}"
                result_summary += f" {web_source.url}"
                logger.log(result_summary)

            # Modify the raw web source text to avoid jinja errors when used in prompt
            search_result.sources = self._postprocess_results(search_result.sources)[:self.limit_per_search]

            # If there is at least one new web source in the result, we were successful
            if len(search_result.sources) > 0:
                self._register_web_sources(search_result.sources)
                return search_result

    def _remove_known_web_sources(self, web_sources: list[WebSource]) -> list[WebSource]:
        """Removes already known websites from the list web_sources."""
        return [r for r in web_sources if r not in self.known_web_sources]

    def _register_web_sources(self, web_sources: list[WebSource]):
        """Adds the provided list of results to the set of known results."""
        self.known_web_sources |= set(web_sources)

    def reset(self):
        """Removes all known web sources and resets the statistics."""
        self.known_web_sources = set()
        self.stats = {s.name: 0 for s in self.search_apis.values()}

    def _postprocess_results(self, results: list[WebSource]) -> list[WebSource]:
        """Modifies the results text to avoid jinja errors when used in prompt."""
        for result in results:
            result.data = self.postprocess_result(result.data, result.query)
        return results

    def postprocess_result(self, result: str, query: Query, filter_relevant: bool = True):
        """Removes all double curly braces to avoid conflicts with Jinja and optionally truncates
        the result text to a maximum length. Also filter the content according to keywords from the query."""
        if filter_relevant and (query.search_mode != SearchMode.REVERSE) and (query.search_mode != SearchMode.IMAGES):
            keywords = re.findall(r'\b\w+\b', query.text.lower()) or query.text
            relevant_content = filter_relevant_sentences(result, keywords)[:10]
            relevant_text = ' '.join(relevant_content)
            result = relevant_text or result

        result = re.sub(r"\{\{.*}}", "", result)
        if self.max_result_len is not None:
            result = result[:self.max_result_len]
        return result

    def _summarize(self, result: SearchResults, **kwargs) -> Optional[MultimediaSnippet]:
        doc = kwargs.get("doc")
        if result:
            for web_source in result.sources:
                self._summarize_single_web_source(web_source, doc)
            return self._summarize_summaries(result, doc)
        else:
            return None

    def _summarize_single_web_source(self, web_source: WebSource, doc: Report):
        prompt = SummarizeResultPrompt(web_source, doc)

        try:
            summary = self.llm.generate(prompt, max_attempts=3)
            if not summary:
                summary = "NONE"
        except APIError as e:
            logger.log(orange(f"APIError: {e} - Skipping the summary for {web_source.url}."))
            logger.log(orange(f"Used prompt:\n{str(prompt)}"))
            summary = "NONE"
        except TemplateSyntaxError as e:
            logger.log(orange(f"TemplateSyntaxError: {e} - Skipping the summary for {web_source.url}."))
            summary = "NONE"
        except ValueError as e:
            logger.log(orange(f"ValueError: {e} - Skipping the summary for {web_source.url}."))
            summary = "NONE"
        except Exception as e:
            logger.log(orange(f"Error while summarizing! {e} - Skipping the summary for {web_source.url}."))
            summary = "NONE"

        web_source.summary = MultimediaSnippet(summary)

        if web_source.is_relevant():
            logger.log("Useful result: " + gray(str(web_source)))

    def _summarize_summaries(self, result: SearchResults, doc: Report) -> Optional[MultimediaSnippet]:
        """Generates a summary, aggregating all relevant information from the
        identified and relevant web sources."""

        summaries = [str(source) for source in result.sources if source.is_relevant()]
        if len(summaries) == 0:  # No relevant web sources
            return None
        elif len(summaries) == 1:
            return MultimediaSnippet(summaries[0])

        # Disable summary of summaries:
        # relevant_sources = "\n\n".join([str(s) for s in result.sources if s.is_relevant()])
        # return MultimediaSnippet(relevant_sources)

        # Prepare the prompt for the LLM
        placeholder_targets = {
            "[SUMMARIES]": str(result),
            "[DOC]": str(doc),
        }
        summarize_prompt = Prompt(placeholder_targets=placeholder_targets,
                                  name="SummarizeSummariesPrompt",
                                  template_file_path="defame/prompts/summarize_summaries.md")

        return MultimediaSnippet(self.llm.generate(summarize_prompt))

    def get_stats(self) -> dict[str, Any]:
        total_searches = np.sum([n for n in self.stats.values()])
        return {
            "Total searches": total_searches,
            "Search engine calls": self.stats,
        }

    def set_date_restriction(self, until: date):
        self.restrict_results_until_date = until

    def set_claim_id(self, claim_id: str):
        # TODO: Test this
        super().set_claim_id(claim_id)
        if "averitec_kb" in self.search_apis:
            kb = self.search_apis["averitec_kb"]
            kb.current_claim_id = claim_id


def filter_relevant_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant_sentences = []
    for sentence in sentences:
        score = sum(1 for word in keywords if word in sentence.lower())
        if score > 0:
            relevant_sentences.append((sentence, score))
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in relevant_sentences]