"""Example for a multimodal fact-check for raw content."""

from ezmm import Image

from defame.fact_checker import FactChecker

fact_checker = FactChecker(llm="gpt_4o", interpret=True, decompose=True)
content = [Image("in/example/us_prison.webp"),
           "look at this! 😱 all because of the democrats!!! 😡 if they continue, half of the US will be ",
           "woke and the other half in jail! Stop the mass incarceration!!!"]
fact_checker.check_content(content)
