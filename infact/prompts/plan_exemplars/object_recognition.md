# Example

## Claim
Image: <image>
Text: "The image shows a protest with many people holding signs."

## Interpretation of the CLAIM
The claim suggests that:
1. There is an image of a protest.
2. Many people in the image are holding signs.

## Factuality Questions
To verify the CLAIM, we need to identify the objects in the image, particularly focusing on signs and people.

ACTIONS:
```
object_recognition(image)
object_recognition(image, "sign")
object_recognition(image, "protesters")
```