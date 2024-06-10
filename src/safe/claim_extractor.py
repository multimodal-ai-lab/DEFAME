from common import utils
from common.console import light_blue
from common.modeling import Model
from eval.logger import EvaluationLogger
from safe import config as safe_config
from safe.prompts.prompt import FilterCheckWorthyPrompt, DecontextualizePrompt, SYMBOL, NOT_SYMBOL
from third_party.factscore.atomic_facts import AtomicFactGenerator


class ClaimExtractor:
    def __init__(self, model: Model, logger: EvaluationLogger = None):
        self.model = model
        self.atomic_fact_generator = AtomicFactGenerator(
            api_key='', gpt3_cache_file='', other_lm=self.model
        )
        self.max_retries = safe_config.max_retries
        self.do_debug = safe_config.debug_safe
        self.logger = logger

    def extract_claims(self, content: str) -> list[str]:
        self.logger.log("Decomposing...", important=True)
        atomic_facts = self.decompose(content)
        for atomic_fact in atomic_facts:
            self.logger.log(light_blue(f"'{atomic_fact}'"))

        self.logger.log("Decontextualizing...")
        atomic_facts_decontextualized = {self.decontextualize(atomic_fact, content) for atomic_fact in atomic_facts}
        for atomic_fact in atomic_facts_decontextualized:
            self.logger.log(light_blue(f"'{atomic_fact}'"))

        self.logger.log("Filtering for unique, check-worthy claims...")
        claims = {claim for claim in atomic_facts_decontextualized if self.is_check_worthy(claim, content)}
    
        for claim in claims:
            self.logger.log(light_blue(f"'{claim}'"), important=True)

        return list(claims)

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
        # TODO: adjust the check-worthiness check & analyze the filter behavior...
        # also make sure multiple claims are not almost the same.

        filter_prompt = FilterCheckWorthyPrompt(atomic_fact, context)

        model_response, answer, num_tries = '', '', 0
        while not answer and num_tries <= self.max_retries:
            model_response = self.model.generate(str(filter_prompt), do_debug=self.do_debug)
            answer = utils.extract_first_square_brackets(model_response)
            answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
            num_tries += 1

        answer = not answer or answer.lower() == SYMBOL.lower()
        return answer
