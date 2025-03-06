from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from defame.common import Image, MultimediaSnippet, Medium


@dataclass(frozen=True)
class SocialMediaPost(MultimediaSnippet):
    """Warning: Do not change attributes after initialization."""
    content: MultimediaSnippet | str | list[str | Medium]  # text and media contained in this post
    platform: str
    post_url: str
    author_username: str

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

    def __post_init__(self):
        if not isinstance(self.content, MultimediaSnippet):
            self.content = MultimediaSnippet(self.content)

        # Compose the MultimediaSnippet
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

        # Add newline to separate from following post content
        text += "\n" + self.content.data

        super().__init__(text)


@dataclass(frozen=True)
class SocialMediaProfile(MultimediaSnippet):
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

    def __post_init__(self):
        # Compose the MultimediaSnippet
        text = f"Profile: @{self.username}"
        if self.display_name:
            text += f" ({self.display_name})"
        if self.is_verified:
            text += " ✓"
        text += f"\n\n{self.bio}\n\n"

        if self.follower_count is not None:
            text += f"Followers: {self.follower_count:,}\n"
        if self.following_count is not None:
            text += f"Following: {self.following_count:,}\n"
        if self.post_count is not None:
            text += f"Posts: {self.post_count:,}\n"

        if self.website:
            text += f"Website: {self.website}\n"

        if self.external_links:
            text += "External Links: " + " ".join(self.external_links) + "\n"

        if self.profile_image:
            text += f"Profile Image: {self.profile_image.reference}\n"

        if self.cover_image:
            text += f"Cover Image: {self.cover_image.reference}\n"

        super().__init__(text)


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
