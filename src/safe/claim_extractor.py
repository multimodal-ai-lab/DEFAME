from common import utils
from common.modeling import Model
from safe import config as safe_config
from third_party.factscore.atomic_facts import AtomicFactGenerator
from common.console import light_blue

SYMBOL = 'Foo'
NOT_SYMBOL = 'Not Foo'
_CONTEXT_PLACEHOLDER = '[CONTEXT]'
_ATOMIC_FACT_PLACEHOLDER = '[ATOMIC FACT]'
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
_REVISE_FORMAT = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
Acorns is a company.

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED STATEMENT:
The subject in the statement "Acorns is a company" is "Acorns". "Acorns" is \
not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "Acorns" is not a vague reference. Thus, the revised statement \
is:
```
Acorns is a company.
```


Example 2:
STATEMENT:
He teaches courses on deep learning.

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Le is also an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

REVISED STATEMENT:
The subject in the statement "He teaches course on deep learning" is "he". \
From the RESPONSE, we can see that this statement comes from the sentence "Le \
is also an adjunct professor at the University of Montreal, where he teaches \
courses on deep learning.", meaning that "he" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
Quoc Le teaches courses on deep learning.
```


Example 3:
STATEMENT:
The television series is called "You're the Worst."

RESPONSE:
Xochitl Gomez began her acting career in theater productions, and she made her \
television debut in 2016 with a guest appearance on the Disney Channel series \
"Raven's Home." She has also appeared in the television series "You're the \
Worst" and "Gentefied."

REVISED STATEMENT:
The subject of the statement "The television series is called "You're the \
Worst."" is "the television series". This is a reference to an unknown entity, \
since it is unclear what television series is "the television series". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "the television series" is a \
vague reference that should be replaced by "the television series that Xochitl \
Gomez appeared in". Thus, the revised response is:
```
The television series that Xochitl Gomez appeared in is called "You're the \
Worst."
```


Example 4:
STATEMENT:
Dean joined Google.

RESPONSE:
Jeff Dean is a Google Senior Fellow and the head of Google AI, leading \
research and development in artificial intelligence. Dean joined Google in \
1999 and has been essential to its continued development in the field.

REVISED STATEMENT:
The subject of the statement "Dean joined Google" is "Dean". From the \
response, we can see that "Dean" is the last name of "Jeff Dean". Therefore \
"Dean" is a non-full name, making it a vague reference. It should be replaced \
by "Jeff Dean", which is the full name. Thus, the revised response is:
```
Jeff Dean joined Google.
```


Your Task:
STATEMENT:
{_ATOMIC_FACT_PLACEHOLDER}

RESPONSE:
{_CONTEXT_PLACEHOLDER}
"""


class ClaimExtractor:
    def __init__(self, model: Model):
        self.model = model
        self.atomic_fact_generator = AtomicFactGenerator(
            api_key='', gpt3_cache_file='', other_lm=self.model
        )
        self.max_retries = safe_config.max_retries
        self.do_debug = safe_config.debug_safe

    def extract_claims(self, content):
        print("Decomposing...")
        atomic_facts = self.decompose(content)
        for atomic_fact in atomic_facts:
            print(light_blue(f"'{atomic_fact}'"))

        print("Decontextualizing...")
        atomic_facts_decontextualized = [self.decontextualize(atomic_fact, content) for atomic_fact in atomic_facts]
        for atomic_fact in atomic_facts_decontextualized:
            print(light_blue(f"'{atomic_fact}'"))

        print("Filtering for check-worthy claims...")
        claims = [claim for claim in atomic_facts_decontextualized if self.is_check_worthy(claim, content)]
        for claim in claims:
            print(light_blue(f"'{claim}'"))

        return claims

    def decompose(self, content: str):
        """Splits up the content into atomic facts."""
        result, _ = self.atomic_fact_generator.run(content)
        atomic_facts = [fact for _, facts in result for fact in facts]
        return atomic_facts

    def decontextualize(self, atomic_fact: str, context: str):
        """Modify the atomic fact to be self-contained."""
        full_prompt = _REVISE_FORMAT.replace(_ATOMIC_FACT_PLACEHOLDER, atomic_fact)
        full_prompt = full_prompt.replace(_CONTEXT_PLACEHOLDER, context)
        full_prompt = utils.strip_string(full_prompt)
        model_response, revised_fact, num_tries = '', '', 0

        while not revised_fact and num_tries <= self.max_retries:
            model_response = self.model.generate(full_prompt, do_debug=self.do_debug)
            revised_fact = utils.extract_first_code_block(
                model_response, ignore_language=True
            )
            num_tries += 1

        return revised_fact or atomic_fact

    def is_check_worthy(self, atomic_fact: str, context: str) -> bool:
        """Identifies whether the given atomic fact is check-worthy."""
        # TODO: adjust the check-worthiness check
        return True

        full_prompt = _RELEVANCE_FORMAT.replace(_ATOMIC_FACT_PLACEHOLDER, atomic_fact)
        # full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
        full_prompt = full_prompt.replace(_CONTEXT_PLACEHOLDER, context)
        full_prompt = utils.strip_string(full_prompt)
        model_response, answer, num_tries = '', '', 0

        while not answer and num_tries <= self.max_retries:
            model_response = self.model.generate(full_prompt, do_debug=self.do_debug)
            answer = utils.extract_first_square_brackets(model_response)
            answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
            num_tries += 1

        answer = not answer or answer.lower() == SYMBOL.lower()
        return answer
