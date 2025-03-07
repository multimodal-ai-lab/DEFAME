import re
from pathlib import Path

from defame.evidence_retrieval.scraping.util import read_urls_from_file
from defame.utils.parsing import get_domain

# These sites don't allow bot access/scraping. Must use a
# proprietary API or a different way to access them.
unsupported_domains_file = Path(__file__).parent / "unsupported_domains.txt"
unsupported_domains = read_urls_from_file(unsupported_domains_file)

# Can be excluded for fairness
fact_checking_urls_file = Path(__file__).parent / "fact_checking_urls.txt"
fact_checking_urls = read_urls_from_file(fact_checking_urls_file)


block_keywords = [
    "captcha",
    "verify you are human",
    "access denied",
    "premium content",
    "403 Forbidden",
    "You have been blocked",
    "Please enable JavaScript",
    "I'm not a robot",
    "Are you a robot?",
    "Are you a human?",
]
unscrapable_urls = [
    "https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html",
    "https://data.news-leader.com/gas-price/",
    "https://www.wlbt.com/2023/03/13/3-years-later-mississippis-/",
    "https://edition.cnn.com/2021/01/11/business/no-fl",
    "https://www.thelugarcenter.org/ourwork-Bipart",
    "https://www.linkedin.com/pulse/senator-kelly-/",
    "http://libres.uncg.edu/ir/list-etd.aspx?styp=ty&bs=master%27s%20thesis&amt=100000",
    "https://www.washingtonpost.com/investigations/coronavirus-testing-denials/2020/03/",
]


def is_fact_checking_site(url: str) -> bool:
    """Check if the URL belongs to a known fact-checking website."""
    # Check if the domain matches any known fact-checking website
    for site in fact_checking_urls:
        if site in url:
            return True
    return False


def is_unsupported_site(url: str) -> bool:
    """Checks if the URL belongs to a known unsupported website."""
    domain = get_domain(url)
    return domain.endswith(".gov") or domain in unsupported_domains or url in unscrapable_urls



def is_relevant_content(content: str) -> bool:
    """Checks if the web scraping result contains relevant content or is blocked by a bot-catcher/paywall."""

    if not content:
        return False

    for keyword in block_keywords:
        if re.search(keyword, content, re.IGNORECASE):
            return False

    return True
