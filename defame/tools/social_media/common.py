from defame.common import Action

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from defame.common import Result, Image, MultimediaSnippet


@dataclass
class SocialMediaPostResult(Result):
    platform: str
    post_url: str
    author_username: str
    post_text: str
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
    text: str = field(init=False)  # This will be assigned in __post_init__

    def __post_init__(self):
        # Create a human-readable text representation
        self.text = f"Post by @{self.author_username}"
        if self.author_display_name:
            self.text += f" ({self.author_display_name})"
        if self.is_verified_author:
            self.text += " ✓"
        self.text += f"\n\n{self.post_text}\n\n"
        
        if self.created_at:
            self.text += f"Posted: {self.created_at.strftime('%B %d, %Y at %H:%M')}\n"
            
        engagement = []
        if self.like_count is not None:
            engagement.append(f"Likes: {self.like_count:,}")
        if self.comment_count is not None:
            engagement.append(f"Comments: {self.comment_count:,}")
        if self.share_count is not None:
            engagement.append(f"Shares: {self.share_count:,}")
            
        if engagement:
            self.text += "Engagement: " + ", ".join(engagement) + "\n"
            
        if self.is_reply and self.reply_to:
            self.text += f"Reply to: @{self.reply_to}\n"
            
        if self.hashtags:
            self.text += "Hashtags: " + " ".join([f"#{tag}" for tag in self.hashtags]) + "\n"
            
        if self.mentions:
            self.text += "Mentions: " + " ".join([f"@{mention}" for mention in self.mentions]) + "\n"
            
        self.text += f"Post URL: {self.post_url}"

        if self.external_links:
            self.text += "\n\n"
            self.text += "External Links: " + " ".join(self.external_links) + "\n"
        
        # Add reference to media if available
        media_references = []
        for i, img in enumerate(self.media):
            media_references.append(img.reference)
        
        if media_references:
            self.text += "\n\nMedia: " + " ".join(media_references)

    def __str__(self):
        return self.text

    def is_useful(self) -> Optional[bool]:
        # Post results are useful if we found a valid post with either text or media
        return (self.post_url != "" and 
                (self.post_text.strip() != "" or len(self.media) > 0))


class RetrieveSocialMediaPost(Action):
    name = "retrieve_post"
    description = "Retrieves a social media post including its text and media content."
    how_to = "Provide the URL of the post."
    format = "retrieve_post(<url:s>)"
    is_multimodal = True
    is_limited = False

    def __init__(self, url: str):
        self.url = url
        self.platform = get_platform(url)

    def __str__(self):
        return f'{self.name}("{self.url}")'

    def __eq__(self, other):
        return isinstance(other, RetrieveSocialMediaPost) and self.url == other.url

    def __hash__(self):
        return hash((self.name, self.url))
    

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