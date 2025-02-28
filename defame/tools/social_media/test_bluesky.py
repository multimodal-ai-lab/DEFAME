"""
Simple script to test the BlueSkyScraper functionality.
This script directly calls the BlueSkyScraper to retrieve a post by URL
and prints the result without needing to integrate with DEFAME.
"""
from defame.tools.social_media.bluesky import bluesky_api


if __name__ == "__main__":    
    # Initialize the scraper

    # Create a test action
    bluesky_url = "https://bsky.app/profile/richardbranson.bsky.social/post/3liykf6dpp225"
    bluesky_url = "https://bsky.app/profile/jamesgunn.bsky.social/post/3lj45gdlhoc2a"
    # result = bluesky_api._retrieve_post(bluesky_url)
    # print(result)

    profile_url = "https://bsky.app/profile/jamesgunn.bsky.social"
    result = bluesky_api.retrieve(profile_url)
    print(result)