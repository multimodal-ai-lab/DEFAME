from datetime import datetime
from io import BytesIO
from typing import Optional

import requests
from PIL import Image as PillowImage
from atproto import Client
from atproto_client.exceptions import RequestErrorBase
from atproto_client.models.common import XrpcError
from ezmm import Image

from defame.common import logger
from .base import SocialMedia
from .common import SocialMediaPost, SocialMediaProfile
from defame.utils.requests import is_image_url


class Bluesky(SocialMedia):
    """Integration for the Bluesky social media platform: https://bsky.app."""
    domains = ["bsky.app"]

    def __init__(self, username: str, password: str):
        super().__init__()
        if not username or not password:
            logger.error("Bluesky username and password must be provided in api_keys.yaml")
            raise ValueError("Bluesky username and password must be provided in api_keys.yaml")

        self.username = username
        self.password = password
        self.n_api_calls = 0
        self.n_errors = 0
        self.authenticated = False
        self.client = Client()
        self._authenticate()

    def _retrieve(self, url: str) -> SocialMediaPost | SocialMediaProfile | None:
        """Retrieve a post from the given URL."""
        if not self.authenticated:
            raise "Bluesky API is not authenticated."

        if "post" in url:
            result = self._retrieve_post(url)
        else:
            result = self._retrieve_profile(url)

        return result

    def _retrieve_post(self, url: str) -> Optional[SocialMediaPost]:
        """Retrieve a post from the given Bluesky URL."""
        # Extract post URI from the URL - Bluesky URLs typically look like:
        # https://bsky.app/profile/username.bsky.social/post/abcdef123
        try:
            # Parse URL to extract components for building the AT URI
            parts = url.split('/')
            if len(parts) < 5 or "bsky.app" not in url:
                raise Exception("Invalid Bluesky URL format.")

            # Find the profile part of the URL
            profile_idx = -1
            for i, part in enumerate(parts):
                if part == "profile":
                    profile_idx = i
                    break

            if profile_idx < 0 or profile_idx + 3 >= len(parts):
                raise Exception("Could not extract profile or post ID.")

            handle = parts[profile_idx + 1]
            post_id = parts[profile_idx + 3]

            # Resolve the handle to a DID
            did = self._resolve_handle(handle)

            # Construct the AT URI
            uri = f"at://{did}/app.bsky.feed.post/{post_id}"

            # Get the post thread from the API
            return self._get_post_thread(uri, url)

        except Exception as e:
            err_msg = error_to_string(e)
            logger.error(f"Error retrieving Bluesky post: {err_msg}")
            self.n_errors += 1

    def _resolve_handle(self, handle: str) -> str:
        """Resolve a handle to a DID."""
        try:
            response = self.client.resolve_handle(handle)
            self.n_api_calls += 1
            return response.did
        except Exception as e:
            err_msg = error_to_string(e)
            logger.error(f"Error resolving handle: {err_msg}")
            self.n_errors += 1
            return handle  # Return the handle itself as fallback

    def _get_post_thread(self, uri: str, original_url: str) -> Optional[SocialMediaPost]:
        """Get the post thread using the Bluesky client."""
        # create an result with default values that will be updated step by step in the following

        try:
            # Get post thread using client method
            thread_response = self.client.get_post_thread(uri=uri, depth=0, parent_height=0)
            self.n_api_calls += 1

            # The thread object contains the post and its context
            thread = thread_response.thread

            # Check if the post exists or is blocked
            if hasattr(thread, 'py_type'):
                # Use dictionary-style access for properties with $ in their name
                thread_type = getattr(thread, 'py_type')
                if thread_type == 'app.bsky.feed.defs#notFoundPost':
                    raise Exception("Post not found")
                if thread_type == 'app.bsky.feed.defs#blockedPost':
                    raise Exception("Post is blocked")

            # Extract post data
            post_view = thread.post
            record = post_view.record

            # Basic post information
            post_text = record.text if hasattr(record, 'text') else ''
            created_at_str = record.created_at[:-1] if hasattr(record, 'created_at') else None

            # Author information
            author = post_view.author
            author_username = author.handle if hasattr(author, 'handle') else ''
            author_display_name = author.display_name if hasattr(author, 'display_name') else ''

            # Viewer info might contain verification status
            is_verified_author = False

            # Engagement metrics
            like_count = post_view.like_count if hasattr(post_view, 'like_count') else 0
            comment_count = post_view.reply_count if hasattr(post_view, 'reply_count') else 0
            share_count = post_view.repost_count if hasattr(post_view, 'repost_count') else 0

            # Extract media (images)
            media = []
            # Check for embedded images in the post
            if hasattr(post_view, 'embed'):
                embed = post_view.embed

                # For image embeds
                if hasattr(embed, 'py_type') and getattr(embed, 'py_type') == 'app.bsky.embed.images#view':
                    for img in embed.images:
                        if hasattr(img, 'fullsize'):
                            try:
                                img_url = img.fullsize
                                if is_image_url(img_url):
                                    # Download the image
                                    image_response = requests.get(img_url, timeout=10)
                                    img_data = PillowImage.open(BytesIO(image_response.content))
                                    media.append(Image(pillow_image=img_data))
                            except Exception as img_err:
                                logger.warning(f"Failed to download image: {str(img_err)}")

            # Extract hashtags and mentions
            hashtags, mentions, external_links = [], [], []
            # Parse facets (rich text features like links, mentions, etc.)
            if hasattr(record, 'facets') and record.facets:
                for facet in record.facets:
                    if hasattr(facet, 'features'):
                        for feature in facet.features:
                            if hasattr(feature, 'py_type'):
                                feature_type = getattr(feature, 'py_type')
                                if feature_type == 'app.bsky.richtext.facet#tag':
                                    hashtags.append(feature.tag if hasattr(feature, 'tag') else '')
                                elif feature_type == 'app.bsky.richtext.facet#mention':
                                    mentions.append(feature.did if hasattr(feature, 'did') else '')
                                elif feature_type == 'app.bsky.richtext.facet#link':
                                    external_links.append(feature.uri)

            # Check if this is a reply
            is_reply, reply_to = False, None
            if hasattr(record, 'reply'):
                is_reply = True
                # Get the parent post's author
                if hasattr(record.reply, 'parent') and hasattr(record.reply.parent, 'uri'):
                    parent_uri = record.reply.parent.uri
                    post_id = parent_uri.split('/')[-1]
                    reply_to_post = self.client.get_posts([parent_uri]).posts[0]
                    self.n_api_calls += 1
                    reply_to_author = reply_to_post.author
                    reply_to = f"https://bsky.app/profile/{reply_to_author.handle}/post/{post_id}"

            # Create the post result
            return SocialMediaPost(
                content=[post_text, *media],
                platform="bluesky",
                post_url=original_url,
                author_username=author_username,
                author_display_name=author_display_name,
                created_at=datetime.fromisoformat(created_at_str) if created_at_str else None,
                like_count=like_count,
                comment_count=comment_count,
                share_count=share_count,
                media=media,
                is_verified_author=is_verified_author,
                is_reply=is_reply,
                reply_to=reply_to,
                hashtags=hashtags,
                mentions=mentions,
                external_links=external_links  # TODO: Integrate external link in post
            )
        except Exception as e:
            err_msg = error_to_string(e)
            logger.error(f"Error getting Bluesky post data: {err_msg}")
            self.n_errors += 1

    def _retrieve_profile(self, url: str) -> Optional[SocialMediaProfile]:
        """Retrieve a profile from the given Bluesky URL."""
        try:
            profile = self.client.get_profile(url.split('/')[-1])
            self.n_api_calls += 1
            profile_image, cover_image = None, None

            if profile.avatar:
                try:
                    image_response = requests.get(profile.avatar, timeout=10)
                    pil_img = PillowImage.open(BytesIO(image_response.content))
                    profile_image = Image(pillow_image=pil_img)
                except Exception as img_err:
                    logger.warning(f"Failed to download profile image: {str(img_err)}")

            if profile.banner:
                try:
                    image_response = requests.get(profile.banner, timeout=10)
                    pil_img = PillowImage.open(BytesIO(image_response.content))
                    cover_image = Image(pillow_image=pil_img)
                except Exception as img_err:
                    logger.warning(f"Failed to download cover image: {str(img_err)}")

            return SocialMediaProfile(
                platform="bluesky",
                profile_url=url,
                username=profile.handle,
                display_name=profile.display_name,
                bio=profile.description,
                is_verified=None,
                follower_count=profile.followers_count,
                following_count=profile.follows_count,
                post_count=profile.posts_count,
                website=None,
                external_links=None,
                profile_image=profile_image,
                cover_image=cover_image,
            )
        except Exception as e:
            err_msg = error_to_string(e)
            logger.error(f"Error retrieving Bluesky profile: {err_msg}")
            self.n_errors += 1

    def _authenticate(self) -> bool:
        """Authenticate with Bluesky using provided credentials."""
        try:
            self.client.login(self.username, self.password)
            self.authenticated = True
            self.n_api_calls += 1
            logger.log(f"✅ Successfully authenticated with Bluesky as {self.username}")
            return True
        except Exception as e:
            logger.error(f"❌ Error authenticating with Bluesky: {str(e)}")
            self.n_errors += 1
            return False

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

# bluesky = Bluesky(api_keys.get("bluesky_username"), api_keys.get("bluesky_password"))

_bluesky_instance = None

def get_bluesky_instance():
    """Get the singleton instance of the Bluesky client."""
    global _bluesky_instance
    if _bluesky_instance is None:
        from defame.config import api_keys
        # Avoid creating the instance if keys are missing, which is useful for tests
        # that don't rely on this specific integration.
        username = api_keys.get("bluesky_username")
        password = api_keys.get("bluesky_password")
        if username and password:
            _bluesky_instance = Bluesky(username, password)
    return _bluesky_instance


bluesky = get_bluesky_instance()


def error_to_string(error: RequestErrorBase | Exception) -> str:
    """Takes an Error object containing a response and prints the contents."""
    if isinstance(error, RequestErrorBase):
        response = error.response
        code = response.status_code
        content = response.content
        if isinstance(content, XrpcError):
            error_type = content.error
            msg = content.message
            return f"Error {code} ({error_type}): {msg}."
        else:
            return f"Error {code}: {content}."
    else:
        return str(error)


if __name__ == "__main__":
    example_url = "https://bsky.app/profile/mrothermel.bsky.social/post/3ldnyqymqgl2c"
    res = bluesky.retrieve(example_url)
    print(res)
