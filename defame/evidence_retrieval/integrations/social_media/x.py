from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, parse_qs

import tweepy

from config.globals import api_keys
from defame.utils.parsing import extract_by_regex
from defame.evidence_retrieval.integrations.search.common import WebSource

USERNAME_REGEX = r"((\w){1,15})"
TWEET_ID_REGEX = r"([0-9]{15,22})"
SEARCH_QUERY_REGEX = r"([\w%\(\)]+)"


class X:
    """TODO: Work in progress.
    The X (Twitter) integration. Requires "Basic" API access to work. For more info, see
    https://developer.x.com/en/docs/twitter-api/getting-started/about-twitter-api#v2-access-level
    "Free" API access does NOT include reading Tweets."""
    name = "x"
    is_free = False
    is_local = False

    def __init__(self):
        self.client = tweepy.Client(bearer_token=api_keys["x_bearer_token"])

    def search(self, query: str, limit: int, start_time: datetime = None) -> list[WebSource]:
        """Searches ALL of X for the given query."""
        if start_time is None:
            start_time = datetime.strptime("26-03-2006", "%d-%m-%Y")
        tweets = self.client.search_all_tweets(
            query=query,
            start_time=start_time,
            max_results=limit
        )
        raise NotImplementedError

    def get_tweet(self, url: str = None, tweet_id: str = None, num_replies: int = 0) -> WebSource:
        assert url is not None or tweet_id is not None
        if url is not None:
            tweet_id = extract_tweet_id_from_url(url)
        tweet = self.client.get_tweet(tweet_id)
        raise NotImplementedError

    def get_user_page(self, url: str = None, username: str = None, num_recent_tweets: int = 0) -> WebSource:
        assert url is not None or username is not None
        if url is not None:
            username = extract_username_from_url(url)
        user_page = self.client.get_user(username=username)
        raise NotImplementedError


def extract_username_from_url(url: str) -> Optional[str]:
    pattern = f"https://twitter\.com/{USERNAME_REGEX}."
    return extract_by_regex(url, pattern)


def extract_tweet_id_from_url(url: str) -> Optional[str]:
    pattern = f"https://twitter\.com/.*/{TWEET_ID_REGEX}.*"
    return extract_by_regex(url, pattern)


def extract_search_query_from_url(url: str) -> Optional[str]:
    if url.startswith("https://twitter.com/search"):
        parsed_url = urlparse(url)
        parsed_url_query = parse_qs(parsed_url.query)
        return parsed_url_query["q"][0]
    else:
        return None
