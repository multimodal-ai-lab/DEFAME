import dataclasses
import re
from typing import Sequence

import numpy as np

from common import shared_config, utils
from common.modeling import Model
from safe import config as safe_config, query_serper
from safe.claim_extractor import ClaimExtractor
from common.label import Label
from common.console import gray, light_blue, bold


SUPPORTED_LABEL = Label.SUPPORTED.value
NOT_SUPPORTED_LABEL = Label.REFUTED.value
# TODO: implement NEI and conflicting evidence
_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
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
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


@dataclasses.dataclass()
class GoogleSearchResult:
    query: str
    result: str


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class FactChecker:
    def __init__(self, model: str | Model = "OPENAI:gpt-3.5-turbo-0125"):
        if isinstance(model, str):
            model = Model(model)
        self.model = model
        self.claim_extractor = ClaimExtractor(model)

        self.debug = safe_config.debug_safe
        self.max_steps = safe_config.max_steps
        self.max_retries = safe_config.max_retries

    def check(self, content: str | Sequence[str]) -> Label:
        """Fact-checks the given content by first extracting all elementary claims and then
        verifying each claim individually. Returns the overall veracity which is true iff
        all elementary claims are true."""

        print(bold(f"Content to be fact-checked: '{light_blue(content)}'"))

        claims = self.claim_extractor.extract_claims(content)

        print(bold("Verifying the claims..."))
        veracities = []
        justifications = []
        for claim in claims:
            veracity, justification = self.verify_claim(claim)
            veracities.append(veracity)
            justifications.append(justification)

        for claim, veracity, justification in zip(claims, veracities, justifications):
            print(bold(f"The claim '{light_blue(claim)}' is {veracity.value}."))
            print(gray(justification))
            print()

        overall_veracity = self.aggregate_predictions(veracities)
        print(bold(f"So, the overall veracity is: {overall_veracity.value}"))

        return overall_veracity

    def aggregate_predictions(self, veracities: Sequence[Label]) -> Label:
        overall_supported = np.all(np.array(veracities) == Label.SUPPORTED)
        overall_veracity = Label.SUPPORTED if overall_supported else Label.REFUTED
        return overall_veracity

    def verify_claim(self, claim: str) -> (Label, str):
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it."""
        search_results = []

        for _ in range(self.max_steps):
            next_search, num_tries = None, 0

            while not next_search and num_tries <= self.max_retries:
                next_search = self.maybe_get_next_search(claim, search_results)
                num_tries += 1

            if next_search is None:
                utils.maybe_print_error('Unsuccessful parsing for `next_search`')
                break
            else:
                search_results.append(next_search)

        search_dicts = {
            'google_searches': [dataclasses.asdict(s) for s in search_results]
        }
        final_answer, num_tries = None, 0

        while not final_answer and num_tries <= self.max_retries:
            num_tries += 1
            final_answer = self.maybe_get_final_answer(
                claim, searches=search_results
            )

        if final_answer is None:
            utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

        predicted_label = Label(final_answer.answer)

        return predicted_label, final_answer.response

    def call_search(self,
                    search_query: str,
                    search_type: str = safe_config.search_type,
                    num_searches: int = safe_config.num_searches,
                    serper_api_key: str = shared_config.serper_api_key,
                    search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
                    ) -> str:
        """Call Google Search to get the search result."""
        search_query += f' {search_postamble}' if search_postamble else ''

        if search_type == 'serper':
            serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
            return serper_searcher.run(search_query, k=num_searches)
        else:
            raise ValueError(f'Unsupported search type: {search_type}')

    def maybe_get_next_search(self,
                              claim: str,
                              past_searches: list[GoogleSearchResult],
                              ) -> GoogleSearchResult | None:
        """Get the next query from the model."""
        knowledge = '\n'.join([s.result for s in past_searches])
        knowledge = 'N/A' if not knowledge else knowledge
        full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, claim)
        full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
        full_prompt = utils.strip_string(full_prompt)
        model_response = self.model.generate(full_prompt, do_debug=self.debug)
        query = utils.extract_first_code_block(model_response, ignore_language=True)

        if model_response and query:
            return GoogleSearchResult(query=query, result=self.call_search(query))

        return None

    def maybe_get_final_answer(self,
                               claim: str,
                               searches: list[GoogleSearchResult],
                               ) -> FinalAnswer | None:
        """Get the final answer from the model."""
        knowledge = '\n'.join([search.result for search in searches])
        full_prompt = _FINAL_ANSWER_FORMAT.replace(
            _STATEMENT_PLACEHOLDER, claim
        )
        full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
        full_prompt = utils.strip_string(full_prompt)
        model_response = self.model.generate(full_prompt, do_debug=self.debug)
        answer = utils.extract_first_square_brackets(model_response)
        answer = re.sub(r'[^\w\s]', '', answer).strip()

        if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
            return FinalAnswer(response=model_response, answer=answer)

        return None
