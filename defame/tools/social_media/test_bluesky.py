"""
Simple script to test the BlueSkyScraper functionality.
This script directly calls the BlueSkyScraper to retrieve a post by URL
and prints the result without needing to integrate with DEFAME.
"""
from defame.tools.social_media.scraper import SocialMediaScraper
from defame.tools.social_media.common import RetrieveSocialMediaPost
from defame.tools.social_media.bluesky import BlueSkyScraper
from config.globals import api_keys


if __name__ == "__main__":    
    # Initialize the scraper
    scraper = SocialMediaScraper(api_keys=api_keys)
    BlueSkyScraper(api_keys=api_keys)  # Register the Bluesky scraper

    # Create a test action
    bluesky_url = "https://bsky.app/profile/richardbranson.bsky.social/post/3liykf6dpp225"
    bluesky_url = "https://bsky.app/profile/jamesgunn.bsky.social/post/3lj45gdlhoc2a"
    action = RetrieveSocialMediaPost(bluesky_url)

    # Perform the action
    evidence = scraper.perform(action)

    # Check the result
    print(evidence)