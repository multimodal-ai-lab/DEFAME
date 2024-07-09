Vague references include but are not limited to:
- Pronouns (e.g., "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. For example, given the response "Titanic is a movie starring Leonardo DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your reasoning. As part of your reasoning, be sure to identify the subjects in the STATEMENT and determine whether they are vague references. If they are vague references, identify the proper entity that they are referring to and be sure to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in a Markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". Some examples have been provided for you to learn how to do this task.


## Example 1
STATEMENT:
Acorns is a company.

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, Jeff Cruttenden, and Mark Dru that provides micro-investing services. The company is headquartered in Irvine, California.

REVISED STATEMENT:
The subject in the statement "Acorns is a company" is "Acorns". "Acorns" is not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" is not further specified in the RESPONSE, so we can assume that it is a full name. Therefore "Acorns" is not a vague reference. Thus, the revised statement is:
```
Acorns is a company.
```


## Example 2
STATEMENT:
He teaches courses on deep learning.

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been working on a variety of deep learning projects. Le is also an adjunct professor at the University of Montreal, where he teaches courses on deep learning.

REVISED STATEMENT:
The subject in the statement "He teaches course on deep learning" is "he". From the RESPONSE, we can see that this statement comes from the sentence "Le is also an adjunct professor at the University of Montreal, where he teaches courses on deep learning.", meaning that "he" refers to "Le". From the RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a non-full name that should be replaced by "Quoc Le." Thus, the revised response is:
```
Quoc Le teaches courses on deep learning.
```


## Example 3
STATEMENT:
The television series is called "You're the Worst."

RESPONSE:
Xochitl Gomez began her acting career in theater productions, and she made her television debut in 2016 with a guest appearance on the Disney Channel series "Raven's Home." She has also appeared in the television series "You're the Worst" and "Gentefied."

REVISED STATEMENT:
The subject of the statement "The television series is called "You're the Worst."" is "the television series". This is a reference to an unknown entity, since it is unclear what television series is "the television series". From the RESPONSE, we can see that the STATEMENT is referring to the television series that Xochitl Gomez appeared in. Thus, "the television series" is a vague reference that should be replaced by "the television series that Xochitl Gomez appeared in". Thus, the revised response is:
```
The television series that Xochitl Gomez appeared in is called "You're the Worst."
```


## Example 4
STATEMENT:
Dean joined Google.

RESPONSE:
Jeff Dean is a Google Senior Fellow and the head of Google AI, leading research and development in artificial intelligence. Dean joined Google in 1999 and has been essential to its continued development in the field.

REVISED STATEMENT:
The subject of the statement "Dean joined Google" is "Dean". From the response, we can see that "Dean" is the last name of "Jeff Dean". Therefore "Dean" is a non-full name, making it a vague reference. It should be replaced by "Jeff Dean", which is the full name. Thus, the revised response is:
```
Jeff Dean joined Google.
```


## Your Task
STATEMENT:
[ATOMIC_FACT]

RESPONSE:
[CONTEXT]