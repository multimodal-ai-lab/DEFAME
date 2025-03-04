from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from defame.common import Result, Image, MultimediaSnippet


@dataclass
class SocialMediaPostResult(Result):
    platform: str
    post_url: str
    author_username: str
    content: MultimediaSnippet  # text and media contained in this post

    created_at: Optional[datetime] = None
    author_display_name: Optional[str] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    share_count: Optional[int] = None
    media: List[Image] = field(default_factory=list)
    is_verified_author: bool = False
    is_reply: bool = False
    reply_to: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)

    def __str__(self):
        # Create a human-readable text representation
        text = f"Post by @{self.author_username}"
        if self.author_display_name:
            text += f" ({self.author_display_name})"
        if self.is_verified_author:
            text += " ✓"

        if self.created_at:
            text += f"\nPosted: {self.created_at.strftime('%B %d, %Y at %H:%M')}\n"

        engagement = []
        if self.like_count is not None:
            engagement.append(f"Likes: {self.like_count:,}")
        if self.comment_count is not None:
            engagement.append(f"Comments: {self.comment_count:,}")
        if self.share_count is not None:
            engagement.append(f"Shares: {self.share_count:,}")

        if engagement:
            text += "Engagement: " + ", ".join(engagement) + "\n"

        text += f"Post URL: {self.post_url}"

        if self.is_reply and self.reply_to:
            text += f"Reply to: {self.reply_to}\n"

        if self.hashtags:
            text += "Hashtags: " + " ".join([f"#{tag}" for tag in self.hashtags]) + "\n"

        if self.mentions:
            text += "Mentions: " + " ".join([f"@{mention}" for mention in self.mentions]) + "\n"

        if self.external_links:
            text += "\n\n"
            text += "External Links: " + " ".join(self.external_links) + "\n"

        # Add reference to media if available
        media_references = []
        for i, img in enumerate(self.media):
            media_references.append(img.reference)

        # Actual post content (text and media)
        text += "\n" + str(self.content)

        return text

    def is_useful(self) -> Optional[bool]:
        # Post results are useful if we found a valid post (verify we have username)
        return self.author_username != ""

@dataclass
class SocialMediaProfileResult(Result):
    platform: str
    profile_url: str
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    is_verified: Optional[bool] = False
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    website: Optional[str] = None
    external_links: Optional[List[str]] = field(default_factory=list)
    profile_image: Optional[Image] = None
    cover_image: Optional[Image] = None
    text: str = field(init=False)

    def __post_init__(self):
        # Create a human-readable text representation
        self.text = f"Profile: @{self.username}"
        if self.display_name:
            self.text += f" ({self.display_name})"
        if self.is_verified:
            self.text += " ✓"
        self.text += f"\n\n{self.bio}\n\n"

        if self.follower_count is not None:
            self.text += f"Followers: {self.follower_count:,}\n"
        if self.following_count is not None:
            self.text += f"Following: {self.following_count:,}\n"
        if self.post_count is not None:
            self.text += f"Posts: {self.post_count:,}\n"

        if self.website:
            self.text += f"Website: {self.website}\n"

        if self.external_links:
            self.text += "External Links: " + " ".join(self.external_links) + "\n"

        if self.profile_image:
            self.text += f"Profile Image: {self.profile_image.reference}\n"

        if self.cover_image:
            self.text += f"Cover Image: {self.cover_image.reference}\n"

    def __str__(self):
        return self.text

    def is_useful(self) -> Optional[bool]:
        # Profile results are useful if we found a valid profile with a username
        return self.username != ""


def get_platform(url: str):
    # Extract platform from URL for routing to correct scraper
    if "x.com" in url or "twitter.com" in url:
        return "x"
    elif "instagram.com" in url:
        return "instagram"
    elif "facebook.com" in url or "fb.com" in url:
        return "facebook"
    elif "tiktok.com" in url:
        return "tiktok"
    elif "bsky.app" in url:
        return "bsky"
    else:
        return "unknown"
