import requests
from datetime import datetime
from io import BytesIO
from PIL import Image as PillowImage
from atproto import Client

from defame.common import Image, logger
from defame.tools.social_media import SocialMediaPostResult, SocialMediaProfileResult
from defame.utils.parsing import is_image_url
from config.globals import api_keys


class BSkyAPI:

    def __init__(self, username: str, password: str):
        if not username or not password:
            logger.error("Bluesky username and password must be provided")
            raise ValueError("Bluesky username and password must be provided")

        self.username = username
        self.password = password
        self.n_api_calls = 0
        self.n_errors = 0
        self.authenticated = False
        self.client = Client()
        self.cache = {}
        self._authenticate()
    
    def supported_platforms(self) -> list[str]:
        """Returns the platforms supported by this scraper."""
        return ["bsky"]
    
    def retrieve(self, url: str) -> SocialMediaPostResult | SocialMediaProfileResult:
        """Retrieve a post from the given URL."""
        if not self.authenticated:
            logger.error("Bluesky API is not authenticated")
            return SocialMediaPostResult.create_error_result(url, "API not authenticated")
        
        if url in self.cache:
            return self.cache[url]
        
        if "bsky.app" not in url:
            return SocialMediaPostResult.create_error_result(url, "Invalid Bluesky URL")
        
        if "post" in url:
            result = self._retrieve_post(url)
        else:
            result = self._retrieve_profile(url)

        self.cache[url] = result
        return result
        
    
    def _retrieve_post(self, url: str) -> SocialMediaPostResult:
        """Retrieve a post from the given Bluesky URL."""
        # Extract post URI from the URL - Bluesky URLs typically look like:
        # https://bsky.app/profile/username.bsky.social/post/abcdef123
        try:
            # Parse URL to extract components for building the AT URI
            parts = url.split('/')
            if len(parts) < 5 or "bsky.app" not in url:
                return SocialMediaPostResult.create_error_result(url, "Invalid Bluesky URL format")
            
            # Find the profile part of the URL
            profile_idx = -1
            for i, part in enumerate(parts):
                if part == "profile":
                    profile_idx = i
                    break
            
            if profile_idx < 0 or profile_idx + 3 >= len(parts):
                return SocialMediaPostResult.create_error_result(url, "Could not extract profile or post ID")
                
            handle = parts[profile_idx + 1]
            post_id = parts[profile_idx + 3]
            
            # Resolve the handle to a DID
            did = self._resolve_handle(handle)
            
            # Construct the AT URI
            uri = f"at://{did}/app.bsky.feed.post/{post_id}"
            
            # Get the post thread from the API
            return self._get_post_thread(uri, url)
            
        except Exception as e:
            logger.error(f"Error retrieving Bluesky post: {str(e)}")
            self.n_errors += 1
            return SocialMediaPostResult.create_error_result(url, f"Error: {str(e)}")
        
    def _resolve_handle(self, handle: str) -> str:
        """Resolve a handle to a DID."""
        try:
            response = self.client.resolve_handle(handle)
            self.n_api_calls += 1
            return response.did
        except Exception as e:
            logger.error(f"Error resolving handle: {str(e)}")
            self.n_errors += 1
            return handle  # Return the handle itself as fallback
    
    def _get_post_thread(self, uri: str, original_url: str) -> SocialMediaPostResult:
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
                    return SocialMediaPostResult.create_error_result(original_url, "Post not found")
                if thread_type == 'app.bsky.feed.defs#blockedPost':
                    return SocialMediaPostResult.create_error_result(original_url, "Post is blocked")
            
            # Extract post data
            post_view = thread.post
            record = post_view.record
            
            # Basic post information
            post_text = record.text if hasattr(record, 'text') else ''
            created_at_str = record.created_at if hasattr(record, 'created_at') else None
            
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
            return SocialMediaPostResult(
                platform="bluesky",
                post_url=original_url,
                author_username=author_username,
                author_display_name=author_display_name,
                post_text=post_text,
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
                external_links=external_links
            )
            
        except Exception as e:
            logger.error(f"Error getting Bluesky post data: {str(e)}")
            self.n_errors += 1
            return SocialMediaPostResult.create_error_result(original_url, f"Error: {str(e)}")
        
    def _retrieve_profile(self, url: str) -> SocialMediaProfileResult:
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

            return SocialMediaProfileResult(
                platform="bluesky",
                profile_url=url,
                username=profile.handle,
                display_name=profile.display_name,
                bio=profile.description,
                is_verified=False,
                follower_count=profile.followers_count,
                following_count=profile.follows_count,
                post_count=profile.posts_count,
                website=None,
                external_links=None,
                profile_image=profile_image,
                cover_image=cover_image,
            )
        except Exception as e:
            logger.error(f"Error retrieving Bluesky profile: {str(e)}")
            self.n_errors += 1
            return SocialMediaProfileResult.create_error_result(url, f"Error: {str(e)}")


    def _authenticate(self) -> bool:
        """Authenticate with Bluesky using provided credentials."""
        try:
            self.client.login(self.username, self.password)
            self.authenticated = True
            self.n_api_calls += 1
            logger.info(f"Successfully authenticated with Bluesky as {self.username}")
            return True
        except Exception as e:
            logger.error(f"Error authenticating with Bluesky: {str(e)}")
            self.n_errors += 1
            return False


bluesky_api = BSkyAPI(api_keys.get("bluesky_username"), api_keys.get("bluesky_password"))

if __name__ == "__main__":
    url = "https://bsky.app/profile/kiddiespeak.bsky.social/post/3lja5miockc2v"
    res = bluesky_api._retrieve_post(url)
    print(res)
