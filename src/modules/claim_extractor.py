from common.claim import Claim
from common.content import Content
from common.modeling import LLM, MLLM
from eval.logger import EvaluationLogger
from prompts.prompt import FilterCheckWorthyPrompt, DecontextualizePrompt, SYMBOL, NOT_SYMBOL
from third_party.factscore import AtomicFactGenerator
from utils.console import light_blue
from utils.parsing import extract_first_square_brackets, extract_first_code_block


class ClaimExtractor:
    def __init__(self, llm: LLM, mllm: MLLM = None,
                 logger: EvaluationLogger = None, do_debug: bool = False):
        self.llm = llm
        self.mllm = mllm
        self.atomic_fact_generator = AtomicFactGenerator(
            api_key='', gpt3_cache_file='', other_lm=self.llm
        )
        self.max_retries = 3
        self.do_debug = do_debug
        self.logger = logger

    def extract_claims(self, content: Content) -> list[Claim]:
        self.logger.log("Starting claim extraction.", important=True)

        self.logger.log("Interpreting...")
        # TODO: Implement

        self.logger.log("Decomposing...")
        atomic_facts = self.decompose(content)
        for atomic_fact in atomic_facts:
            self.logger.log(light_blue(f"'{atomic_fact}'"))

        self.logger.log("Decontextualizing...")
        claims = {self.decontextualize(atomic_fact, content) for atomic_fact in atomic_facts}
        for claim in claims:
            self.logger.log(light_blue(f"'{claim}'"))

        self.logger.log("Filtering for unique, check-worthy claims...")
        claims = [claim for claim in claims if self.is_check_worthy(claim, content)]

        for claim in claims:
            self.logger.log(light_blue(f"'{claim}'"), important=True)

        return claims

    def decompose(self, content: Content):
        """Splits up the content into atomic facts."""
        interpretation = content.interpretation
        result, _ = self.atomic_fact_generator.run(interpretation)  # TODO: Implement handling the interpretation
        atomic_facts = [fact for _, facts in result for fact in facts]
        return atomic_facts

    def decontextualize(self, atomic_fact: str, context: Content) -> Claim:
        """Modify the atomic fact to be self-contained."""
        decontextualize_prompt = DecontextualizePrompt(atomic_fact, context)

        model_response, revised_fact, num_tries = '', '', 0
        while not revised_fact and num_tries <= self.max_retries:
            model_response = self.llm.generate(str(decontextualize_prompt), do_debug=self.do_debug)
            revised_fact = extract_first_code_block(
                model_response, ignore_language=True
            )
            num_tries += 1

        return Claim(revised_fact or atomic_fact)

    def is_check_worthy(self, claim: Claim, context: Content) -> bool:
        """Identifies whether the given atomic fact is check-worthy."""
        # TODO: adjust the check-worthiness check & analyze the filter behavior...
        # also make sure multiple claims are not almost the same.

        filter_prompt = FilterCheckWorthyPrompt(claim, context)

        model_response, answer, num_tries = '', '', 0
        while not answer and num_tries <= self.max_retries:
            model_response = self.llm.generate(str(filter_prompt), do_debug=self.do_debug)
            answer = extract_first_square_brackets(model_response)
            answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
            num_tries += 1

        answer = not answer or answer.lower() == SYMBOL.lower()
        return answer
