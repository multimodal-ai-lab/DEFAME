# Example

## Claim
Text: "<image:4> shows a protest with many people holding signs."

## Interpretation of the CLAIM
The claim suggests that:
1. There is an image of a protest.
2. Many people in the image are holding signs.

## Factuality Questions
To verify the CLAIM, we need to identify the objects in the image, particularly focusing on signs and people.

ACTIONS:
```
detect_objects(<image:4>)
detect_objects(<image:4>, "sign")
detect_objects(<image:4>, "protesters")
```