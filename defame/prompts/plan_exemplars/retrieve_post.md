# Example: retrieve_post()

## Claim
Text: "According to a social media post shared on Bluesky by Richard Branson, he is selling Virgin Galactic to focus on AI development <image:k>"

## Reasoning
To verify this claim, I need to examine the actual post made by Richard Branson. First, I'll use web search to find the mentioned post on Bluesky.

## Actions
```
web_search("Richard Branson Virgin Galactic sale Bluesky post")
```

## Results
The search found a news article mentioning a recent Bluesky post by Richard Branson at the URL: https://bsky.app/profile/richardbranson.bsky.social/post/3liykf6dpp225

## Actions
```
retrieve_post("https://bsky.app/profile/richardbranson.bsky.social/post/3liykf6dpp225")
```

## Results
Retrieved a social media post from Bluesky.

## Reasoning
The retrieved post from NASA's official Bluesky account shows an image from the James Webb Space Telescope, but it only shows distant galaxies and stars. There is no mention or evidence of alien civilizations in the post. The claim misrepresents the content of NASA's actual social media post.