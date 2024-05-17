from safe.prompts.common import STATEMENT_PLACEHOLDER, KNOWLEDGE_PLACEHOLDER

NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does \
not support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query.
4. Your query should aim to obtain new information that does not appear \
in the KNOWLEDGE. 
5. Format your final query by putting it in a markdown code block. \

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""

NEXT_SEARCH_FORMAT_OPEN_SOURCE = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE.
2. Your goal is to try to find evidence that either supports or does not \
support the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does NOT appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT. Remember, the query should yield new, additional information that is not in KNOWLEDGE.
5. Do NOT include any websites, do NOT use the token 'site:' and only output the pure natural query that I would use in Google Search.
6. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""
