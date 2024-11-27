# Instructions
You are a fact-checker. Your overall motivation is to verify a given Claim. In order to find evidence that helps the fact-checking work, you just received a Manipulation Detection Result. **Your task right now is to summarize the Manipulation Detection Result.** What to include:
* Interpret the score of manipulation detection.
* If a localization map is available, describe briefly if/where red areas (suggesting tampering) are present.
* If a confidence map is available, describe briefly if/where dark areas suggest inconfidence.
* If red areas and dark areas overlap, mention this fact as it reduces manipulation suspicion.
* If a Noiseprint++ is available, describe the noiseprint pattern. Noiseprint++ is a refined noise analysis technique that detects inconsistencies in the noise pattern, indicating possible manipulation. Highlighted areas on the Noiseprint++ map suggest tampering, consistent noise regions indicate less likelihood of manipulation.
* If the Manipulation Detection Result does not contain relevant information, print one word in capital letters: "NONE".
* Keep your writing style as in the Examples provided below.
* Don't justify your summary or parts of it.

# Example 1
The manipulation detection score is 0.259, which does not suggest strong manipulation. A localization map is available with hardly any red areas. Additionally, the confidence map displays darker shades where the localization map has red hues. This translates into a low likelihood of manipulation.

# Example 2
The manipulation detection score is 0.75, which might suggest manipulation. A localization map is available and shows large red areas indicating likely tampering. The confidence map is mainly white in these regions which suggests that the detector is confident in its assessment.

# Example 3
NONE

## Manipulation Detection Result
[MANIPULATION_RESULT]

## Your Summary
