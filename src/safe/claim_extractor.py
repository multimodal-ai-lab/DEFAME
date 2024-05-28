from common import utils
from common.modeling import Model
from safe import config as safe_config
from third_party.factscore.atomic_facts import AtomicFactGenerator
from common.console import light_blue
from safe.prompts.prompt import FilterCheckWorthyPrompt, DecontextualizePrompt, SYMBOL, NOT_SYMBOL
from eval.logging import print_log
from logging import Logger
from typing import Optional


class ClaimExtractor:
    def __init__(self, model: Model):
        self.model = model
        self.atomic_fact_generator = AtomicFactGenerator(
            api_key='', gpt3_cache_file='', other_lm=self.model
        )
        self.max_retries = safe_config.max_retries
        self.do_debug = safe_config.debug_safe

    def extract_claims(self, content, verbose=False, logger: Optional[Logger] = None,):
        if verbose:
            print("Decomposing...")
        if logger:
            print_log(logger, "Decomposing...")
        atomic_facts = self.decompose(content)
        for atomic_fact in atomic_facts:
            if verbose:
                print(light_blue(f"'{atomic_fact}'"))
            if logger:
                print_log(logger, f"'{atomic_fact}'")

        if verbose:
            print("Decontextualizing...")
        if logger:
            print_log(logger, "Decontextualizing...")
        atomic_facts_decontextualized = [self.decontextualize(atomic_fact, content) for atomic_fact in atomic_facts]
        for atomic_fact in atomic_facts_decontextualized:
            if verbose:
                print(light_blue(f"'{atomic_fact}'"))
            if logger:
                print_log(logger, f"'{atomic_fact}'")
        if verbose:
            print("Filtering for check-worthy claims...")
        if logger:
            print_log(logger, "Filtering for check-worthy claims...")
        claims = [claim for claim in atomic_facts_decontextualized if self.is_check_worthy(claim, content)]
        for claim in claims:
            if verbose:
                print(light_blue(f"'{claim}'"))
            if logger:
                print_log(logger, f"'{claim}'")

        return claims

    def decompose(self, content: str):
        """Splits up the content into atomic facts."""
        result, _ = self.atomic_fact_generator.run(content)
        atomic_facts = [fact for _, facts in result for fact in facts]
        return atomic_facts

    def decontextualize(self, atomic_fact: str, context: str):
        """Modify the atomic fact to be self-contained."""
        decontextualize_prompt = DecontextualizePrompt(atomic_fact, context)

        model_response, revised_fact, num_tries = '', '', 0
        while not revised_fact and num_tries <= self.max_retries:
            model_response = self.model.generate(str(decontextualize_prompt), do_debug=self.do_debug)
            revised_fact = utils.extract_first_code_block(
                model_response, ignore_language=True
            )
            num_tries += 1

        return revised_fact or atomic_fact

    def is_check_worthy(self, atomic_fact: str, context: str) -> bool:
        """Identifies whether the given atomic fact is check-worthy."""
        # TODO: adjust the check-worthiness check

        filter_prompt = FilterCheckWorthyPrompt(atomic_fact, context)

        model_response, answer, num_tries = '', '', 0
        while not answer and num_tries <= self.max_retries:
            model_response = self.model.generate(str(filter_prompt), do_debug=self.do_debug)
            answer = utils.extract_first_square_brackets(model_response)
            answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
            num_tries += 1

        answer = not answer or answer.lower() == SYMBOL.lower()
        return answer
