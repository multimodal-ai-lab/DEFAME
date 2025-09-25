from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

from ezmm import MultimodalSequence, Image

from defame.common import Results
from ....analysis.stance import Stance


class SearchMode(Enum):
    SEARCH = "search"
    NEWS = "news"
    PLACES = "places"
    IMAGES = "images"
    REVERSE = "reverse"  # Reverse Image Search (RIS)


@dataclass
class Query:
    text: Optional[str] = None
    image: Optional[Image] = None
    search_mode: Optional[SearchMode] = None
    limit: Optional[int] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def __post_init__(self):
        assert self.text or self.image, "Query must have at least one of 'text' or 'image'."

    def has_text(self) -> bool:
        return self.text is not None

    def has_image(self) -> bool:
        return self.image is not None

    @property
    def start_time(self) -> datetime:
        return datetime.combine(self.start_date, datetime.min.time()) if self.start_date else None

    @property
    def end_time(self) -> datetime:
        return datetime.combine(self.end_date, datetime.max.time()) if self.end_date else None

    def __eq__(self, other):
        return isinstance(other, Query) and (
            self.text == other.text and
            self.image == other.image and
            self.limit == other.limit and
            self.start_date == other.start_date and
            self.end_date == other.end_date and
            self.search_mode == other.search_mode
        )

    def __hash__(self):
        return hash((
            self.text,
            self.image,
            self.limit,
            self.start_date,
            self.end_date,
            self.search_mode,
        ))


@dataclass
class Source:
    """A source of information. For example, a web page or an excerpt
     of a local knowledge base. Each source must be clearly identifiable
     (and ideally also retrievable) by its reference."""
    reference: str  # e.g. URL
    content: MultimodalSequence = None  # the contained raw information, may require lazy scrape
    takeaways: MultimodalSequence = None  # an optional summary of the content's relevant info

    def is_loaded(self) -> bool:
        return self.content is not None

    def is_relevant(self) -> Optional[bool]:
        """Returns True if the summary contains information helpful for the fact-check."""
        if self.takeaways is None:
            return None
        elif str(self.takeaways) == "":
            return False
        else:
            return "NONE" not in str(self.takeaways)

    def _get_content_str(self):
        if self.is_relevant():
            return f"Takeaways: {str(self.takeaways)}"
        elif self.is_loaded():
            return f"Content: {str(self.content)}"
        else:
            return "⚠️ Content not yet loaded."

    def __str__(self):
        """Uses the summary if available, otherwise the raw content."""
        text = f"Source {self.reference}\n"
        return text + self._get_content_str()

    def __eq__(self, other):
        return isinstance(other, Source) and self.reference == other.reference

    def __hash__(self):
        return hash(self.reference)

    def __repr__(self):
        return f"Source(reference='{self.reference}')"


@dataclass(eq=False, unsafe_hash=False)
class WebSource:
    """A generic container for web content retrieved from any source."""
    reference: str  # The URL or identifier of the source
    content: MultimodalSequence  # The main content
    title: Optional[str] = None  # The title of the page or content
    preview: Optional[str] = None  # A short preview of the content
    stance: Optional[Stance] = None # The stance of the content towards a claim
    credibility: Optional[float] = None # The credibility score of the content

    def __str__(self):
        return f"WebSource(reference='{self.reference}', title='{self.title}')"
    
    def __hash__(self):
        """Hash based on reference URL for deduplication purposes."""
        return hash(self.reference)
    
    def __eq__(self, other):
        """Equality based on reference URL for deduplication purposes."""
        if not isinstance(other, WebSource):
            return False
        return self.reference == other.reference
    
    def is_loaded(self):
        """Check if the source content has been loaded (scraped)."""
        return self.content is not None and len(str(self.content).strip()) > 0
    
    @property
    def url(self) -> str:
        """Compatibility property for scraper - returns the reference URL."""
        return self.reference
    
    def is_relevant(self) -> bool:
        """Returns True if the source contains relevant information for fact-checking."""
        # A source is relevant if it has takeaways that don't indicate "NONE"
        if hasattr(self, 'takeaways') and self.takeaways is not None:
            return "NONE" not in str(self.takeaways)
        # Otherwise, it's relevant if it has loaded content
        return self.is_loaded()


@dataclass
class SearchResults(Results):
    """A list of sources."""
    sources: list[Source]
    query: Query  # the query that resulted in these sources

    @property
    def n_sources(self):
        return len(self.sources)

    def __str__(self):
        if self.n_sources == 0:
            return "No search results found."
        else:
            return "**Search Results**\n\n" + "\n\n".join(map(str, self.sources))

    def __repr__(self):
        return f"SearchResults(n_sources={len(self.sources)}, sources={self.sources})"
