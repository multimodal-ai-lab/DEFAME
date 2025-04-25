import re
from datetime import datetime, timedelta, date
from typing import Any, Optional

from ezmm import Image, MultimodalSequence
from jinja2.exceptions import TemplateSyntaxError
from openai import APIError

from config.globals import api_keys
from defame.common import Report, Prompt, logger, Action
from defame.evidence_retrieval import scraper
from defame.evidence_retrieval.integrations.search import SearchResults, SearchPlatform, PLATFORMS, KnowledgeBase
from defame.evidence_retrieval.integrations.search.common import Query, SearchMode, Source, WebSource
from defame.evidence_retrieval.tools.tool import Tool
from defame.prompts.prompts import SummarizeSourcePrompt
from defame.utils.console import gray


class Search(Action):
    """Runs a search on the specified platform to retrieve helpful sources. Useful
    to find new knowledge. Some platforms also support images, e.g.,
    Reverse Image Search (RIS), or
    search modes (like 'news', 'places'), and additional parameters like date limits.
    If a platform does not support some of the parameters, they will be ignored.
    If you run multiple search queries, vary them."""
    name = "search"

    platform: SearchPlatform
    query: Query

    def __init__(self,
                 query: str = None,
                 image: str = None,
                 platform: str = "google",
                 mode: str = "search",
                 limit: int = None,
                 start_date: str = None,
                 end_date: str = None):
        """
        @param query: The textual search query. At least one of `query` or `image` must
            be set.
        @param image: The reference of an image. Use this if you want to perform Reverse
            Image Search (RIS). RIS is helpful to find sources that contain the same or
            similar images. If you also provide `query`, the query will be treated as
            additional context, constraining the search results respectively.
        @param platform: The platform/engine to run the query on. Choose from the
            available platforms below.
        @param mode: The search mode or category. Choose from
            `search` for standard, open search (default),
            `images` for retrieving images for a given text query (useful for verifying
                claims that feature visuals),
            `news` for searching (recent) news articles,
            `places` for searching places.
        @param limit: The maximum number of search results to retrieve.
        @param start_date: Returns search results on or after this date. Use ISO format.
        @param end_date: Returns search results before or on this date. Use ISO format.
        """
        self._save_parameters(locals())

        try:
            self.platform = PLATFORMS[platform]
        except KeyError:
            logger.warning(f"Platform {platform} is not available. Defaulting to Google.")
            self.platform = PLATFORMS["google"]

        image = Image(reference=image) if image else None

        try:
            mode = SearchMode(mode) if mode else None
        except ValueError:
            mode = None

        try:
            start_date = date.fromisoformat(start_date) if start_date else None
        except ValueError:
            start_date = None

        try:
            end_date = date.fromisoformat(end_date) if end_date else None
        except ValueError:
            end_date = None

        self.query = Query(text=query, image=image, search_mode=mode, limit=limit,
                           start_date=start_date, end_date=end_date)

    def __eq__(self, other):
        return isinstance(other, Search) and self.query == other.query and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.query))


class Searcher(Tool):
    """Searches the specified platform (Google, Wikipedia, ...) for useful sources."""
    # TODO: Rank or annotate the websites according to their credibility, like MUSE
    name = "searcher"
    platforms: list[SearchPlatform]

    n_retrieved_results: int
    n_unique_retrieved_results: int

    def __init__(self,
                 search_config: dict[str, dict] = None,
                 limit_per_search: int = 5,
                 max_result_len: int = None,  # chars
                 extract_sentences: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.limit_per_search = limit_per_search
        self.max_result_len = max_result_len  # chars
        self.extract_sentences = extract_sentences
        self.restrict_results_before_time: Optional[datetime] = None  # date restriction for all search actions

        self.platforms = self._initialize_platforms(search_config)
        self.known_sources: set[Source] = set()

        self.actions = self._define_actions()

        self.reset()

    def _initialize_platforms(self, search_config: Optional[dict]) -> list[SearchPlatform]:
        if search_config is None:
            search_config = self._get_default_search_config()

        platforms = []
        for platform, kwargs in search_config.items():
            if kwargs is None:
                kwargs = {}
            if platform == "averitec_kb":
                kwargs["device"] = self.device
            platform_cls = PLATFORMS[platform]
            platform = platform_cls(max_search_results=self.limit_per_search, **kwargs)
            platforms.append(platform)

        return platforms

    def _get_default_search_config(self):
        if api_keys["serper_api_key"]:
            return {"google": {}}
        else:
            logger.warning("No Serper API key (needed for Google) provided. Falling back to DuckDuckGo.")
            return {"duckduckgo": {}}

    def _define_actions(self) -> list[type[Action]]:
        """Adds a list of the available search platforms to the Search
        action class which will be used in the LLM prompt.."""
        platforms_info = "Available search platforms:"
        for platform in self.platforms:
            platforms_info += f"\n`{platform.name}`: {platform.description}"
        Search.additional_info = platforms_info
        return [Search]

    def _perform(self, action: Search) -> Optional[SearchResults]:
        """Validates the search query (by enforcing potential restrictions)
        and runs it."""
        query = action.query

        # Set the strictest specified end date
        if self.restrict_results_before_time is not None:
            max_date = self.restrict_results_before_time.date() - timedelta(days=1)
            if query.end_date is not None:
                query.end_date = min(query.end_date, max_date)
            else:
                query.end_date = max_date

        # Set the strictest search limit
        if self.limit_per_search is not None:
            if query.limit is not None:
                query.limit = min(query.limit, self.limit_per_search)
            else:
                query.limit = self.limit_per_search

        # Ensure the given platform is available
        platform = self.get_platform(action.platform.name)
        if not platform:
            platform = self.platforms[0]
            logger.warning(f"Platform {action.platform.name} is not initialized/allowed. "
                           f"Defaulting to {platform.name}.")

        # Run the query
        return self._search(platform, query)

    def _search(self, platform: SearchPlatform, query: Query) -> Optional[SearchResults]:
        """Executes the given search query on the given platform and processes the results.
        Removes known results."""

        # Run search and retrieve sources
        results = platform.search(query)
        sources = results.sources[:self.limit_per_search]
        self.n_retrieved_results += len(sources)

        # Remove known sources
        sources = self._remove_known_sources(sources)
        self.n_unique_retrieved_results += len(sources)

        # Log search results
        if len(sources) > 0:
            logger.log(f"Got {len(sources)} new source(s):")
            logger.log("\n".join([s.reference for s in sources]))
        else:
            logger.log("No new sources found.")

        # Scrape the pages of the results
        sources_to_scrape = [s for s in sources if isinstance(s, WebSource)]
        scraper.scrape_sources(sources_to_scrape)

        # Modify the raw source text to avoid jinja errors when used in prompt
        self._postprocess_sources(sources, query)
        self._register_sources(sources)

        if len(sources) > 0:
            results.sources = sources
            return results

    def _remove_known_sources(self, sources: list[Source]) -> list[Source]:
        """Removes already known sources from the list `sources`."""
        return [r for r in sources if r not in self.known_sources]

    def _register_sources(self, sources: list[Source]):
        """Adds the provided list of sources to the set of known sources."""
        self.known_sources |= set(sources)

    def reset(self):
        """Removes all known web sources and resets the search platforms."""
        self.known_sources = set()
        self.n_retrieved_results = 0
        self.n_unique_retrieved_results = 0
        for platform in self.platforms:
            platform.reset()

    def _postprocess_sources(self, sources: list[Source], query: Query) -> None:
        for source in sources:
            if source.is_loaded():
                processed = self._postprocess_single_source(str(source.content), query)
                source.content = MultimodalSequence(processed)

    def _postprocess_single_source(self, content: str, query: Query) -> str:
        """Prepares the result contents before LLM processing:
        1. Optionally extracts relevant sentences from the result text using keywords
            from the query.
        2. Removes all double curly braces to avoid conflicts with Jinja.
        3. Optionally truncates the result text to a maximum length."""
        if self.extract_sentences:
            keywords = re.findall(r'\b\w+\b', query.text.lower()) or query.text
            relevant_content = extract_relevant_sentences(content, keywords)[:10]
            relevant_text = ' '.join(relevant_content)
            content = relevant_text or content

        content = re.sub(r"\{\{.*}}", "", content)

        if self.max_result_len is not None:
            content = content[:self.max_result_len]

        return content

    def _summarize(self, results: SearchResults, doc: Report = None) -> Optional[MultimodalSequence]:
        assert doc is not None
        if results:
            for source in results.sources:
                self._summarize_single_source(source, doc)
            return self._summarize_summaries(results, doc)
        else:
            return None

    def _summarize_single_source(self, source: Source, doc: Report):
        prompt = SummarizeSourcePrompt(source, doc)

        try:
            summary = self.llm.generate(prompt, max_attempts=3)
            if not summary:
                summary = "NONE"
        except APIError as e:
            logger.info(f"APIError: {e} - Skipping the summary for {source}.")
            logger.log(f"Used prompt:\n{str(prompt)}")
            summary = "NONE"
        except TemplateSyntaxError as e:
            logger.info(f"TemplateSyntaxError: {e} - Skipping the summary for {source}.")
            summary = "NONE"
        except ValueError as e:
            logger.warning(f"ValueError: {e} - Skipping the summary for {source}.")
            summary = "NONE"
        except Exception as e:
            logger.log(f"Error while summarizing! {e} - Skipping the summary for {source}.")
            summary = "NONE"

        source.takeaways = MultimodalSequence(summary)

        if source.is_relevant():
            logger.log("Useful source: " + gray(str(source)))

    def _summarize_summaries(self, result: SearchResults, doc: Report) -> Optional[MultimodalSequence]:
        """Generates a summary, aggregating all relevant information from the
        identified and relevant sources."""

        summaries = [str(source) for source in result.sources if source.is_relevant()]
        if len(summaries) == 0:  # No relevant sources
            return None
        elif len(summaries) == 1:
            # No further summarization needed as we have only one source
            return MultimodalSequence(summaries[0])

        # Disable summary of summaries:
        # relevant_sources = "\n\n".join([str(s) for s in result.sources if s.is_relevant()])
        # return MultimodalSequence(relevant_sources)

        # Prepare the prompt for the LLM
        placeholder_targets = {
            "[SUMMARIES]": str(result),
            "[DOC]": str(doc),
        }
        summarize_prompt = Prompt(placeholder_targets=placeholder_targets,
                                  name="SummarizeSummariesPrompt",
                                  template_file_path="defame/prompts/summarize_summaries.md")

        return MultimodalSequence(self.llm.generate(summarize_prompt))

    def get_stats(self) -> dict[str, Any]:
        return {
            "Total searches": sum([platform.n_searches for platform in self.platforms]),
            "Platform stats": {platform.name: platform.stats for platform in self.platforms},
        }

    def get_platform(self, name: str) -> Optional[SearchPlatform]:
        for platform in self.platforms:
            if platform.name == name:
                return platform

    def set_time_restriction(self, before: Optional[datetime]):
        self.restrict_results_before_time = before

    def set_claim_id(self, claim_id: str):
        super().set_claim_id(claim_id)
        kb = self.get_platform(KnowledgeBase.name)
        if kb:
            kb.current_claim_id = int(claim_id)


def extract_relevant_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant_sentences = []
    for sentence in sentences:
        score = sum(1 for word in keywords if word in sentence.lower())
        if score > 0:
            relevant_sentences.append((sentence, score))
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in relevant_sentences]
