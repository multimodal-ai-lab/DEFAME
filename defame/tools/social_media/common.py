from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from defame.common import Result, Image


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
            self.text += f"Reply to: {self.reply_to}\n"
            
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
        # Post results are useful if we found a valid post (verify we have username)
        return self.author_username != ""
    
    @staticmethod
    def create_error_result(url: str, error_message: str) -> 'SocialMediaPostResult':
        """Create a SocialMediaPostResult that indicates an error occurred."""
        return SocialMediaPostResult(
            platform="unknown",
            post_url=url,
            author_username="",
            post_text=f"Error: {error_message}",
        )


@dataclass
class SocialMediaProfileResult(Result):
    platform: str
    profile_url: str
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    is_verified: bool = False
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    website: Optional[str] = None
    external_links: List[str] = field(default_factory=list)
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
    
    @staticmethod
    def create_error_result(url: str, error_message: str) -> 'SocialMediaProfileResult':
        """Create a SocialMediaProfileResult that indicates an error occurred."""
        return SocialMediaProfileResult(
            platform="unknown",
            profile_url=url,
            username="",
            bio=f"Error: {error_message}",
        )



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