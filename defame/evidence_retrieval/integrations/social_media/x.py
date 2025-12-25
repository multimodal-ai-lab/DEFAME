from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests
import tweepy
from ezmm import Image

from config.globals import api_keys
from defame.evidence_retrieval.integrations import SocialMediaPost
from defame.evidence_retrieval.integrations.social_media.common import SocialMediaPostMetadata
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

    def keyword_search(self, query: str, start_time: datetime, timeframe: int = 60, post_limit=50) -> list[SocialMediaPost]:
        """
        Searches ALL of X for the given query and returns them as a list of social media posts.
        :param query: Keywords to search for.
        :param post_limit: Maximum number of posts to search for.
        :param start_time: The oldest possible timestamp of a post to be included
        :param timeframe: The duration of the search window in minutes.
        :return: SocialMediaPost objects with properties 'platform', 'author', 'url' and 'images'.
        """

        matching_posts = []

        if post_limit < 10: # requirement from Twitter API
            post_limit = 10
        if post_limit > 100:
            post_limit = 100

        end_time = start_time + timedelta(minutes=timeframe)
        end_time -= timedelta(seconds=15) # API requires small delay

        tweet_fields = [
            "author_id",
            "text",
            "created_at",
            "attachments",
            "referenced_tweets",
        ]
        user_fields = ["name", "username"]
        media_fields = ["url", "preview_image_url", "variants"]
        expansions = ["attachments.media_keys", "referenced_tweets.id", "author_id"]

        if start_time > end_time:
            start_time, end_time = end_time, start_time
        response = self.client.search_recent_tweets(
            query=query,
            max_results=post_limit,
            tweet_fields=tweet_fields,
            media_fields=media_fields,
            expansions=expansions,
            user_fields=user_fields,
            start_time=start_time,
            end_time=end_time,
        )

        if response.data is None:
            return matching_posts  # No posts found

        lookup = {t.id: t for t in response.includes["tweets"]}
        media_dict = {m["media_key"]: m for m in response.includes.get("media", [])}
        users = {u["id"]: u for u in response.includes["users"]}
        for post in response.data:
            original = post


            if ( # The message of retweets is not completely in the message itself. Therefor, needs to be retrieved via extension individually.
                post.referenced_tweets
                and post.referenced_tweets[0].type == "retweeted"
            ):

                original_id = post.referenced_tweets[0].id
                original_tweet = lookup[original_id]
                post = original_tweet # Use the original post instead of the retweet

            msg = post.text.replace("\n", " ")


            # retrieve images from post
            images = []
            if post.attachments and (img_keys := post.attachments.get("media_keys")):
                for key in img_keys:
                    media = media_dict.get(key)
                    if media and media["type"] == "photo":
                        img_response = requests.get(media["url"])
                        img_response.raise_for_status()
                        images.append(Image(binary_data=img_response.content))

            username = users.get(original.author_id, "Unknown")

            metadata = SocialMediaPostMetadata(
                platform="X",
                author_username = username,
                post_url = f"https://x.com/{username}/status/{post.id}",
                media=images, # empty list equals default values
            )
            matching_posts.append(SocialMediaPost(metadata=metadata, message=msg))
        return matching_posts

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
