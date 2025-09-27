import re
from datetime import datetime, timedelta, date
from typing import Any, Optional

from ezmm import Image, MultimodalSequence
from jinja2.exceptions import TemplateSyntaxError
from openai import APIError

from config.globals import api_keys
from defame.common import Report, Prompt, logger, Action, Evidence
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


class SearchSocialReddit(Action):
    """Search Reddit specifically for posts related to a claim using Reddit's API.
    This will find relevant Reddit discussions and register their URLs for analysis 
    by the Reddit tool."""
    name = "search_social_reddit"

    def __init__(self, query: str, limit: int = 10):
        """
        @param query: The textual search query to find Reddit posts
        @param limit: Maximum number of Reddit posts to find (default: 10)
        """
        self._save_parameters(locals())
        self.query = query
        self.limit = limit

    def __eq__(self, other):
        return isinstance(other, SearchSocialReddit) and self.query == other.query and self.limit == other.limit

    def __hash__(self):
        return hash((self.name, self.query, self.limit))


class SearchSocialX(Action):
    """Search X (Twitter) specifically for posts related to a claim using X's API.
    This will find relevant X/Twitter discussions and register their URLs for analysis 
    by the X tool."""
    name = "search_social_x"

    def __init__(self, query: str, limit: int = 10):
        """
        @param query: The textual search query to find X/Twitter posts
        @param limit: Maximum number of X/Twitter posts to find (default: 10)
        """
        self._save_parameters(locals())
        self.query = query
        self.limit = limit

    def __eq__(self, other):
        return isinstance(other, SearchSocialX) and self.query == other.query and self.limit == other.limit

    def __hash__(self):
        return hash((self.name, self.query, self.limit))


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
        
        # Add information about social media search capabilities
        social_info = """

Social Media Search Actions:
`search_social_reddit`: Search Reddit specifically using their API to find relevant posts and discussions
`search_social_x`: Search X (Twitter) specifically using their API to find relevant posts and discussions

Note: Social media searches will find posts and register their URLs for analysis by the respective social media tools."""
        
        Search.additional_info = platforms_info + social_info
        
        # Return all available search actions
        return [Search, SearchSocialReddit, SearchSocialX]

    def perform(self, action: Action, summarize: bool = True, **kwargs) -> Evidence:
        """Override perform to handle social media search actions."""
        assert type(action) in self.actions, f"Forbidden action: {action}"
        
        # Handle social media search actions
        if isinstance(action, SearchSocialReddit):
            result = self._perform_social_reddit_search(action)
        elif isinstance(action, SearchSocialX):
            result = self._perform_social_x_search(action)
        else:
            # Handle regular search actions
            result = self._perform(action)
        
        summary = self._summarize(result, **kwargs) if summarize else None
        return Evidence(result, action, takeaways=summary)

    def _perform_social_reddit_search(self, action: SearchSocialReddit) -> Optional[SearchResults]:
        """Execute Reddit-specific search using scrapeMM Reddit integration."""
        try:
            logger.info(f"Performing Reddit search for: '{action.query}' (limit: {action.limit})")
            
            # Run the social media search which will register URLs automatically
            search_results = self.search_and_analyze_social_media(action.query, max_results_per_platform=action.limit)
            reddit_urls = search_results.get("reddit", [])
            
            # Create WebSource objects for the found URLs
            sources = []
            from ..integrations.search.common import WebSource
            
            for url in reddit_urls:
                source = WebSource(
                    reference=url,
                    content=None,  # Will be loaded when the Reddit tool processes it
                    title=f"Reddit discussion - {action.query[:50]}...",
                    preview=f"Reddit post found via API search for '{action.query}'"
                )
                sources.append(source)
            
            # Create a SearchResults object
            from ..integrations.search import SearchResults
            result = SearchResults(sources=sources, query=action.query)
            
            logger.info(f"Reddit search completed: found {len(reddit_urls)} posts")
            return result
            
        except Exception as e:
            logger.error(f"Error in Reddit search: {e}")
            return None

    def _perform_social_x_search(self, action: SearchSocialX) -> Optional[SearchResults]:
        """Execute X/Twitter-specific search using scrapeMM X integration."""
        try:
            logger.info(f"Performing X search for: '{action.query}' (limit: {action.limit})")
            
            # Run the social media search which will register URLs automatically
            search_results = self.search_and_analyze_social_media(action.query, max_results_per_platform=action.limit)
            x_urls = search_results.get("x", [])
            
            # Create WebSource objects for the found URLs
            sources = []
            from ..integrations.search.common import WebSource
            
            for url in x_urls:
                source = WebSource(
                    reference=url,
                    content=None,  # Will be loaded when the X tool processes it
                    title=f"X discussion - {action.query[:50]}...",
                    preview=f"X/Twitter post found via API search for '{action.query}'"
                )
                sources.append(source)
            
            # Create a SearchResults object
            from ..integrations.search import SearchResults
            result = SearchResults(sources=sources, query=action.query)
            
            logger.info(f"X search completed: found {len(x_urls)} posts")
            return result
            
        except Exception as e:
            logger.error(f"Error in X search: {e}")
            return None

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
        Removes known results. Also includes social media search results."""

        # Run search and retrieve sources
        results = platform.search(query)
        sources = results.sources[:self.limit_per_search]
        self.n_retrieved_results += len(sources)
        
        # Also search social media platforms if we have a text query
        if query.text:
            social_media_sources = self._search_social_media_platforms(query.text)
            sources.extend(social_media_sources)
            logger.log(f"Added {len(social_media_sources)} social media sources to search results")

        # Remove known sources
        sources = self._remove_known_sources(sources)
        self.n_unique_retrieved_results += len(sources)

        # Log search results
        if len(sources) > 0:
            logger.log(f"Got {len(sources)} new source(s):")
            logger.log("\n".join([s.reference for s in sources]))
            
            # Register social media URLs for use by social media tools
            self._register_social_media_urls(sources)
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
    
    def _register_social_media_urls(self, sources: list[Source]):
        """Register Reddit and X/Twitter URLs found in search results for use by social media tools."""
        import re
        from ..shared_urls import add_reddit_urls, add_x_urls
        
        reddit_urls = []
        x_urls = []
        
        for source in sources:
            url = source.reference
            # Check for Reddit URLs
            if "reddit.com" in url and re.match(r'https://(?:www\.)?reddit\.com/r/[^/]+/comments/[^/\s]+', url):
                reddit_urls.append(url)
                print(f"🔍 Registering Reddit URL from search: {url}")
            
            # Check for X/Twitter URLs
            if ("twitter.com" in url or "x.com" in url) and re.match(r'https://(?:www\.)?(?:twitter|x)\.com/[^/]+/status/\d+', url):
                x_urls.append(url)
                print(f"🔍 Registering X URL from search: {url}")
        
        if reddit_urls:
            add_reddit_urls(reddit_urls)
            print(f"🔍 Added {len(reddit_urls)} Reddit URLs to registry")
        
        if x_urls:
            add_x_urls(x_urls)
            print(f"🔍 Added {len(x_urls)} X URLs to registry")

    async def search_social_media_platforms(self, query: str, max_results_per_platform: int = 10) -> dict[str, list[str]]:
        """Search Reddit and X platforms for posts related to the query.
        
        Args:
            query: The search query (typically a claim or part of a claim)
            max_results_per_platform: Maximum results per platform
            
        Returns:
            Dict with platform names as keys and lists of URLs as values
        """
        results = {
            "reddit": [],
            "x": []
        }
        
        # Import the scrapeMM integrations
        from scrapemm.integrations.reddit import Reddit as ScrapeMM_Reddit
        from scrapemm.integrations.x import X as ScrapeMM_X
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Search Reddit
            try:
                reddit_scraper = ScrapeMM_Reddit()
                if reddit_scraper.connected:
                    reddit_urls = await reddit_scraper.search(query, session, max_results_per_platform)
                    results["reddit"] = reddit_urls
                    logger.info(f"Found {len(reddit_urls)} Reddit posts for query: {query}")
                else:
                    logger.warning("Reddit integration not connected, skipping Reddit search")
            except Exception as e:
                logger.error(f"Error searching Reddit: {e}")
            
            # Search X (Twitter)
            try:
                x_scraper = ScrapeMM_X()
                if x_scraper.connected:
                    # X API requires minimum 10 results, maximum 100
                    x_limit = max(10, min(100, max_results_per_platform))
                    x_urls = await x_scraper.search(query, session, x_limit)
                    results["x"] = x_urls[:max_results_per_platform]  # Limit to requested amount
                    logger.info(f"Found {len(x_urls)} X posts for query: {query}")
                else:
                    logger.warning("X integration not connected, skipping X search")
            except Exception as e:
                logger.error(f"Error searching X: {e}")
        
        return results

    def _extract_social_keywords(self, claim_text: str) -> str:
        """Extract keywords from claim text optimized for social media search."""
        try:
            from defame.prompts.prompts import ExtractSocialKeywordsPrompt
            prompt = ExtractSocialKeywordsPrompt(claim_text)
            
            logger.info(f"🔍 KEYWORD EXTRACTION - Input claim: '{claim_text}'")
            
            # Use the LLM if available, otherwise fallback to simple extraction
            if hasattr(self, 'llm') and self.llm:
                logger.info("🤖 Using LLM for keyword extraction")
                logger.info(f"📝 LLM PROMPT:\n{str(prompt)}")
                
                response = self.llm.generate(prompt)
                
                logger.info(f"📤 LLM RESPONSE (type={type(response)}): {response}")
                
                if response and isinstance(response, str):
                    # Clean up the response and use it directly
                    keywords = response.strip()
                    # Remove any extra formatting that might be present
                    keywords = keywords.replace('Keywords:', '').replace('**', '').strip()
                    logger.info(f"✅ LLM keywords after cleanup: '{keywords}'")
                elif response and isinstance(response, dict):
                    # Handle dict response (shouldn't happen with new prompt but just in case)
                    keywords = str(response.get('keywords', response.get('response', ''))).strip()
                    logger.info(f"✅ LLM keywords from dict: '{keywords}'")
                else:
                    logger.warning("❌ Failed to extract keywords via LLM, using fallback")
                    keywords = self._fallback_keyword_extraction(claim_text)
            else:
                logger.info("⚠️ No LLM available, using fallback keyword extraction")
                keywords = self._fallback_keyword_extraction(claim_text)
            
            # Ensure we have reasonable keywords (not empty and not too long)
            if not keywords or len(keywords.split()) > 8:
                logger.warning(f"⚠️ LLM keywords invalid (empty or too long: {len(keywords.split()) if keywords else 0} words), using fallback")
                keywords = self._fallback_keyword_extraction(claim_text)
                
            logger.info(f"🎯 FINAL KEYWORDS: '{keywords}' (from claim: '{claim_text[:100]}...')")
            return keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}, using fallback")
            return self._fallback_keyword_extraction(claim_text)
    
    def _fallback_keyword_extraction(self, claim_text: str) -> str:
        """Simple fallback keyword extraction without LLM."""
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'said', 'says', 
            'claim', 'claims', 'according', 'reports', 'allegedly', 'apparently', 'shows'
        }
        
        # Extract potential keywords with better patterns
        # Look for capitalized words (likely names, places) - preserve as found
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', claim_text)
        
        # Look for dates/years
        dates = re.findall(r'\b(?:20\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2})\b', claim_text)
        
        # Extract meaningful words (3+ chars, not stop words)
        words = re.findall(r'\b\w{3,}\b', claim_text.lower())
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Remove duplicates while preserving order and avoid case conflicts
        seen = set()
        keywords = []
        
        # Add proper nouns first (max 2)
        for noun in proper_nouns[:2]:
            if noun.lower() not in seen:
                keywords.append(noun)
                seen.add(noun.lower())
        
        # Add dates (max 1)
        for date in dates[:1]:
            if date.lower() not in seen:
                keywords.append(date)
                seen.add(date.lower())
        
        # Add remaining meaningful words until we have 5 total
        for word in meaningful_words:
            if len(keywords) >= 5:
                break
            if word not in seen:
                keywords.append(word)
                seen.add(word)
        
        return ' '.join(keywords[:5])

    def search_and_analyze_social_media(self, claim_text: str, max_results_per_platform: int = 10):
        """Search social media platforms and register URLs for analysis by respective tools.
        
        This method:
        1. Extracts keywords from the claim for better social media search
        2. Searches Reddit and X for posts related to the keywords
        3. Registers found URLs in the shared registry
        4. The planner will then use SearchReddit/SearchX actions to analyze these posts
        
        Args:
            claim_text: The claim to search for
            max_results_per_platform: Maximum results per platform
        """
        
        # Extract keywords optimized for social media search
        search_keywords = self._extract_social_keywords(claim_text)
        logger.info(f"Social media search using keywords: '{search_keywords}'")
        
        # Use keywords for the actual search
        actual_query = search_keywords if search_keywords else claim_text
        
        # Run the async search with extracted keywords
        import asyncio
        try:
            # Handle potential event loop issues
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.search_social_media_platforms(actual_query, max_results_per_platform))
                    search_results = future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                search_results = asyncio.run(self.search_social_media_platforms(actual_query, max_results_per_platform))
                
            # Register found URLs for use by social media tools
            from ..shared_urls import add_reddit_urls, add_x_urls
            
            if search_results["reddit"]:
                add_reddit_urls(search_results["reddit"])
                logger.info(f"Registered {len(search_results['reddit'])} Reddit URLs for analysis")
            
            if search_results["x"]:
                add_x_urls(search_results["x"])
                logger.info(f"Registered {len(search_results['x'])} X URLs for analysis")
                
            return search_results
            
        except Exception as e:
            logger.error(f"Error in social media search: {e}")
            return {"reddit": [], "x": []}

    def _search_social_media_platforms(self, query_text: str) -> list[Source]:
        """Search social media platforms and return Source objects for found URLs.
        
        This method extracts keywords from the query and searches Reddit and X APIs 
        for content related to those keywords, creating WebSource objects that can 
        be integrated with regular search results.
        
        Args:
            query_text: The search query text
            
        Returns:
            List of WebSource objects for found social media URLs
        """
        from ..integrations.search.common import WebSource
        
        social_sources = []
        
        # Extract keywords for better social media search
        search_keywords = self._extract_social_keywords(query_text)
        logger.debug(f"Regular search social media integration using keywords: '{search_keywords}'")
        
        # Get URLs from social media search with graceful error handling
        try:
            logger.debug(f"Attempting social media search for query: {query_text}")
            # Use keyword-based search instead of raw query
            social_media_results = self.search_and_analyze_social_media(search_keywords or query_text, max_results_per_platform=5)
            
            reddit_count = len(social_media_results.get("reddit", []))
            x_count = len(social_media_results.get("x", []))
            logger.debug(f"Social media search completed: {reddit_count} Reddit, {x_count} X posts")
            
            # Convert Reddit URLs to WebSource objects
            for reddit_url in social_media_results.get("reddit", []):
                try:
                    reddit_source = WebSource(
                        reference=reddit_url,
                        content=None,  # Will be loaded later during scraping
                        title=f"Reddit discussion - {search_keywords[:50] if search_keywords else query_text[:50]}...",
                        preview=f"Social media discussion from Reddit API search (keywords: {search_keywords})"
                    )
                    social_sources.append(reddit_source)
                    logger.debug(f"Added Reddit source: {reddit_url}")
                except Exception as e:
                    logger.warning(f"Error creating Reddit source for {reddit_url}: {e}")
            
            # Convert X URLs to WebSource objects  
            for x_url in social_media_results.get("x", []):
                try:
                    x_source = WebSource(
                        reference=x_url,
                        content=None,  # Will be loaded later during scraping
                        title=f"X discussion - {search_keywords[:50] if search_keywords else query_text[:50]}...",
                        preview=f"Social media discussion from X API search (keywords: {search_keywords})"
                    )
                    social_sources.append(x_source)
                    logger.debug(f"Added X source: {x_url}")
                except Exception as e:
                    logger.warning(f"Error creating X source for {x_url}: {e}")
            
            if social_sources:
                logger.info(f"✅ Social media search found {len(social_sources)} additional sources")
            else:
                logger.debug("ℹ️ No additional social media sources found")
                    
        except Exception as e:
            logger.info(f"ℹ️ Social media search unavailable: {e}")
            logger.debug("Continuing with web search results only")
        
        return social_sources

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
