from typing import Optional

from ezmm import MultimodalSequence

from defame.utils.parsing import get_domain


class RetrievalIntegration:
    """An integration (external API or similar) that is able to retrieve the contents
    for a given URL belonging to a specific domain. Maintains a cache."""

    domains: list[str]

    def __init__(self):
        self.cache = {}

    def retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Returns the contents at the URL."""
        assert get_domain(url) in self.domains
        if url in self.cache:
            return self.cache[url]
        else:
            result = self._retrieve(url)
            self.cache[url] = result
            return result

    def _retrieve(self, url: str) -> Optional[MultimodalSequence]:
        raise NotImplementedError
