"""
Simple script to test the Bluesky integration.
"""
from defame.evidence_retrieval.integrations.social_media.bluesky import bluesky


def test_posts():
    urls_to_test = [
        "https://bsky.app/profile/richardbranson.bsky.social/post/3liykf6dpp225",
        "https://bsky.app/profile/jamesgunn.bsky.social/post/3lj45gdlhoc2a",
    ]

    for url in urls_to_test:
        result = bluesky.retrieve(url)
        print(result)


def test_profile():
    urls_to_test = [
        "https://bsky.app/profile/jamesgunn.bsky.social"
    ]

    for url in urls_to_test:
        result = bluesky.retrieve(url)
        print(result)
