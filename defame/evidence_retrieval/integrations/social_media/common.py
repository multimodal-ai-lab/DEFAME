from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from ezmm import Image, MultimodalSequence


@dataclass(frozen=True)
class SocialMediaPostMetadata:
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



class SocialMediaPost(MultimodalSequence):
    metadata: SocialMediaPostMetadata

    def __init__(self, *args, metadata: SocialMediaPostMetadata):
        self.metadata = metadata

        # Compose the sequence
        text = f"Post by @{metadata.author_username}"
        if metadata.author_display_name:
            text += f" ({metadata.author_display_name})"
        if metadata.is_verified_author:
            text += " ✓"

        if metadata.created_at:
            text += f"\nPosted: {metadata.created_at.strftime('%B %d, %Y at %H:%M')}\n"

        engagement = []
        if metadata.like_count is not None:
            engagement.append(f"Likes: {metadata.like_count:,}")
        if metadata.comment_count is not None:
            engagement.append(f"Comments: {metadata.comment_count:,}")
        if metadata.share_count is not None:
            engagement.append(f"Shares: {metadata.share_count:,}")

        if engagement:
            text += "Engagement: " + ", ".join(engagement) + "\n"

        text += f"Post URL: {metadata.post_url}"

        if metadata.is_reply and metadata.reply_to:
            text += f"Reply to: {metadata.reply_to}\n"

        if metadata.hashtags:
            text += "Hashtags: " + " ".join([f"#{tag}" for tag in metadata.hashtags]) + "\n"

        if metadata.mentions:
            text += "Mentions: " + " ".join([f"@{mention}" for mention in metadata.mentions]) + "\n"

        if metadata.external_links:
            text += "\n\n"
            text += "External Links: " + " ".join(metadata.external_links) + "\n"

        # Add reference to media if available
        media_references = []
        for i, img in enumerate(metadata.media):
            media_references.append(img.reference)

        # Add newline to separate from following post content
        text += "\n"

        super().__init__(text, *args)


@dataclass(frozen=True)
class SocialMediaProfile(MultimodalSequence):
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
        # Compose the sequence
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
