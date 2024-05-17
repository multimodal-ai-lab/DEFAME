from common.label import Label
from safe.prompts.common import STATEMENT_PLACEHOLDER, KNOWLEDGE_PLACEHOLDER

FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{Label.SUPPORTED.value}" or \
"{Label.REFUTED.value}". Wrap your final answer in square brackets.

KNOWLEDGE:
{KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{STATEMENT_PLACEHOLDER}
"""
