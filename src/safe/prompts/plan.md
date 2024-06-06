# Instructions
1. You are a fact-checker. Your overall motivation is to verify a given CLAIM.
2. Under "Your Task", you can re-read the part of the fact-checking work you have already done, it might be empty. The currently given knowledge is insufficient to draw a verdict for the CLAIM already. Hence, we need to find more evidence.
3. Your task right now is to propose the next action (or actions) to take in order to find more evidence.
4. Under VALID_ACTIONS you can see which actions are available, including a short description. No other actions are possible at this moment. 
5. Under REASONING, you think step-by-step. There you should analyze what information is missing. You are encouraged to pose explicit questions that ask for the missing information. Moreover, under REASONING, you can also analyze the already available knowledge from "Your Task" and bring evidence peaces together, if necessary.
6. Finally, under NEXT_ACTIONS, you're going to propose the next actions to take (as available in VALID_ACTIONS). Format your proposed actions by putting them into a Markdown code block.
7. You're provided with one example that shows how it is done.

VALID_ACTIONS:
`WEB_SEARCH`
Description: run a search on Google or DuckDuckGO
Format: `WEB_SEARCH: your web search query goes here`

`WIKI_LOOKUP`
Description: look up something on Wikipedia
Format: `WIKI_LOOKUP: your wiki search query goes here`


# Example
CLAIM:
“The combined population of Wyoming, North Dakota and South Dakota is smaller than the total US prison population.”

REASONING:
In order to verify this claim, we need to compare the current population count of the three states with the total number of US inmates. How many people live in Wyoming, North Dakota, and South Dakota? How many people are imprisoned in the US? To answer these questions, we need to search the web for current numbers.

ACTIONS:
```
WIKI_LOOKUP: Wyoming
WIKI_LOOKUP: North Dakota
WIKI_LOOKUP: South Dakota
WEB_SEARCH: total US prison population
```

RESULTS:
Wyoming has a population count of 576,851 in 2020 ([Wikipedia](https://en.wikipedia.org/wiki/Wyoming))
North Dakota has a population count of less than 780,000 ([Wikipedia](https://en.wikipedia.org/wiki/North_Dakota))
South Dakota has a population count of 909,824 in 2022 ([Wikipedia](https://en.wikipedia.org/wiki/South_Dakota))
As of May 2024, there are 158,703 people imprisoned on federal level in the US ([Federal Bureau of Prisons](https://www.bop.gov/mobile/about/population_statistics.jsp)), however there are also state-level inmates

REASONING:
The total US inmates figure is incomplete because we only know the number of federal-level inmates but not the number of state-level inmates. How many people are imprisoned on state-level in the US? Maybe there is some statistic which combines all the different inmate numbers on state and federal level. To find out more, we search the web for some statistics and reports that aggregate all the numbers.

NEXT_ACTIONS:
```
WEB_SEARCH: total aggregated US prison population state and federal level
WEB_SEARCH: state-level US prison population
WEB_SEARCH: number of US inmates report
```


# Your Task
[DOC]

REASONING:
