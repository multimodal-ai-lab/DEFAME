# Example
This example shows you how it is done.

## Claim
Text: “The combined population of Wyoming, North Dakota and South Dakota is smaller than the total US prison population.”
Claim date: Aug 23, 2024

## Reasoning
In order to verify this claim, we need to compare the current population count of the three states with the total number of US inmates. How many people live in Wyoming, North Dakota, and South Dakota? How many people are imprisoned in the US? To answer these questions, we need to search the web for current numbers.

## Actions
```
wiki_lookup("Wyoming")
wiki_lookup("North Dakota")
wiki_lookup("South Dakota")
web_search("total US prison population")
```

## Results
4 useful new results found.

## Reasoning
Wyoming has a population count of 576,851 in 2020 ([Wikipedia](https://en.wikipedia.org/wiki/Wyoming)).
North Dakota has a population count of less than 780,000 ([Wikipedia](https://en.wikipedia.org/wiki/North_Dakota)).
South Dakota has a population count of 909,824 in 2022 ([Wikipedia](https://en.wikipedia.org/wiki/South_Dakota)).
As of May 2024, there are 158,703 people imprisoned on federal level in the US ([Federal Bureau of Prisons](https://www.bop.gov/mobile/about/population_statistics.jsp)), however there are also state-level inmates

The total US inmates figure is incomplete because we only know the number of federal-level inmates but not the number of state-level inmates. How many people are imprisoned on state-level in the US? Maybe there is some statistic which combines all the different inmate numbers on state and federal level. To find out more, we search the web for some statistics and reports that aggregate all the numbers.

## Actions
```
web_search("total aggregated US prison population state and federal level")
web_search("state-level US prison population")
web_search("number of US inmates report")
```
