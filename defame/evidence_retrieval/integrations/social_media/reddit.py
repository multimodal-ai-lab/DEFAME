import praw
from ezmm import Image

import requests
import requests.auth
from datetime import datetime, timezone, timedelta
from config.globals import api_keys
from defame.evidence_retrieval.integrations import SocialMediaPost
from defame.evidence_retrieval.integrations.social_media.common import SocialMediaPostMetadata


class Reddit:
    """
    The Reddit API Integration. See https://praw.readthedocs.io/en/stable/ for more information.
    """
    name = "reddit"

    def __init__(self):
        self.client = praw.Reddit(
            client_id=api_keys["reddit_client_id"],
            client_secret=api_keys["reddit_client_secret"],
            user_agent=api_keys["reddit_token"],
            password=api_keys["reddit_pwd"],
            username=api_keys["reddit_username"]
        )

    def keyword_search(
        self,
        keywords: str,
        start_time: datetime,
        timeframe: int = 60,
        post_limit: int = 50,
        sort_order: str = "new",
        subreddit: list[str] = None,
    ) -> list[SocialMediaPost]:
        """
        Retrieve reddit posts from Reddit.
        :param keywords: The keywords to search for.
        :param post_limit: The maximum number of posts to return. Default 10
        :param start_time: Oldest a post can be in minutes to still be included in minutes. Default to last hour
        :param timeframe: The timeframe to use. Default 60 minutes
        :param sort_order: The sort order to use. Options: "relevance", "hot", "top", "new", "comments". Default "new"
        :param subreddit: The subreddits to use. Default: all subreddits
        :return: The list of Reddit posts that match the criteria.
        """

        matching_posts = []

        if subreddit is None:
            subreddit = ["all"]

        query = ", ".join(keywords)  # ggf auf einfach array
        res = []

        for s in subreddit:
            sr = self.client.subreddit(s)
            for post in sr.search(
                query, limit=post_limit, sort=sort_order, time_filter="month"
            ):
                res.append(post)
        res.sort(key=lambda p: p.created_utc, reverse=True)
        res = list(
            filter(
                lambda p: start_time.timestamp()
                <= p.created_utc
                <= (start_time+timedelta(minutes=timeframe)).timestamp(),
                res,
            )
        )
        res = res[: min(post_limit, len(res))]

        for post in res:
            author = post.author.name
            date = datetime.fromtimestamp(int(post.created_utc), timezone.utc)
            message = post.title + ((" | " + post.selftext) if post.selftext else "")

            img_urls = []
            if hasattr(post, "preview") and post.preview is not None:
                for item in post.preview["images"]:
                    img_urls.append(item["source"]["url"])
            if hasattr(post, "media_metadata") and post.media_metadata is not None:
                img_urls.extend(
                    [item["s"]["u"] for item in post.media_metadata.values()]
                )

            images = []
            for url in img_urls:
                response = requests.get(url)
                img_bytes = response.content
                images.append(Image(binary_data=img_bytes))

            matching_posts.append(
                SocialMediaPost(message=message, metadata=SocialMediaPostMetadata(platform="Reddit", post_url=post.url, author_username=author, created_at=date, media=images))
            )

        return matching_posts
