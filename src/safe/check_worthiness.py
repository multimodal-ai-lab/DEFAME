"""Identifies whether a given atomic fact is check-worthy."""

from typing import Any

from common import modeling, utils
from safe import config as safe_config
from safe.decontextualize import _CONTEXT_PLACEHOLDER, _ATOMIC_FACT_PLACEHOLDER, revise_fact

SYMBOL = 'Foo'
NOT_SYMBOL = 'Not Foo'

# TODO: Update this for actual check-worthiness

_RELEVANCE_FORMAT = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Quoc is well-respected by many \
of his peers, such as Geoffrey Hinton, who is an adjunct professor at the \
University of Montreal and teaches courses on deep learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. The phrase "Quoc is well-respected by many of his peers, such \
as Geoffrey Hinton" from the RESPONSE shows that the relationship between Quoc \
Le and Geoffrey Hinton is that they are peers. For this reason, the subjects \
Quoc Le and Geoffrey Hinton are [{SYMBOL}].


Example 2:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Geoffrey Hinton is an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and Geoffrey Hinton is. Thus, the subjects Quoc Le and \
Geoffrey Hinton are [{NOT_SYMBOL}].


Your Task:
QUESTION:
[PROMPT]

RESPONSE:
{_CONTEXT_PLACEHOLDER}

STATEMENT:
{_ATOMIC_FACT_PLACEHOLDER}
"""


def determine_check_worthiness_single(
        atomic_fact: str,
        context: str,
        model: modeling.Model,
        do_debug: bool = safe_config.debug_safe,
        max_retries: int = safe_config.max_retries,
) -> tuple[str, bool]:
    """Check if the atomic fact is relevant for answering the prompt."""
    full_prompt = _RELEVANCE_FORMAT.replace(_ATOMIC_FACT_PLACEHOLDER, atomic_fact)
    # full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
    full_prompt = full_prompt.replace(_CONTEXT_PLACEHOLDER, context)
    full_prompt = utils.strip_string(full_prompt)
    model_response, answer, num_tries = '', '', 0

    while not answer and num_tries <= max_retries:
        model_response = model.generate(full_prompt, do_debug=do_debug)
        answer = utils.extract_first_square_brackets(model_response)
        answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
        num_tries += 1

    answer = not answer or answer.lower() == SYMBOL.lower()
    return model_response, answer  # if no parsed answer, assume relevant


def determine_check_worthiness(
        atomic_facts: list[str],
        context: str,
        model: modeling.Model
):
    check_worthy = []
    for atomic_fact in atomic_facts:
        response, fact_is_check_worthy = determine_check_worthiness_single(atomic_fact, context, model)
        check_worthy.append(fact_is_check_worthy)
    return check_worthy
