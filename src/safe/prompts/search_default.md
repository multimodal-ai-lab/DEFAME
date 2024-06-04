Instructions:
1. You have been given a STATEMENT and KNOWLEDGE points.
2. Your goal is to find evidence that either supports or refutes the given STATEMENT.
3. To do this, you are allowed to issue ONE search query.
4. Your query should aim to obtain new information that does not appear in the KNOWLEDGE. 
5. Avoid proposing a query that is listed under PAST_QUERIES. If a query in PAST_QUERIES is marked with "Bad Query", propose a substantially different query.
6. Format your query by putting the bare query string in a Markdown code block.
7. Think step-by-step (REASONING) what information we need to know and what a good query might be to retrieve that information.
8. Your task is to do this for the KNOWLEDGE, PAST_QUERIES, and STATEMENT under "Your Task". Some examples have been provided for you to learn how to do this task.


## Example 1
KNOWLEDGE:
N/A

PAST_QUERIES:
N/A

STATEMENT:
Trump Administration claimed songwriter Billie Eilish Is Destroying Our Country In Leaked Documents

REASONING:
The STATEMENT is a meta claim (claim about a claim): In particular, the STATEMENT implies the Trump Administration to accuse Billie Ellish to destroy the US. Therefore, we need to ask:
- Do these "leaked documents" exist at all?
- If yes, do they contain the accusation as stated?
- Are they in fact from the Trump Administration?
Since there are no PAST_QUERIES, our first query aims to find resources related to the STATEMENT in general.

NEXT_QUERY:
```trump administration aboout billie ellish destroying US```


## Example 2
KNOWLEDGE:
Wyoming has a population of about 580k people.
North Dakota's population counts 770k individuals.
There are 900k inhabitants in South Dakota.

PAST_QUERIES:
Wyoming population size
North Dakota population size
South Dakota population size

STATEMENT:
Wyoming, North Dakota, and South Dakota have a combined population smaller than the US prison population.

REASONING:
The STATEMENT does connect multiple facts: The populations of three different US states and the total US prison population. Here, we need to know:
- How many people live in Wyoming?
- How many people live in North Dakota?
- How many people live in South Dakota?
- How many people are in prison in the US?
- How do the numbers compare?
Thanks to the previous search queries, we already know the states' population sizes. Hence, we continue with searching for the total number of US inmates.

NEXT_QUERY:
```total number of US inmates```


## Your Task
KNOWLEDGE:
[KNOWLEDGE]

PAST_QUERIES:
[PAST_QUERIES]

STATEMENT:
[STATEMENT]

REASONING:
