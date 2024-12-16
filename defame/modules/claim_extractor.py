from defame.common import logger, Model, Content, Claim
from defame.prompts.prompts import SYMBOL, NOT_SYMBOL, DecontextualizePrompt, FilterCheckWorthyPrompt, InterpretPrompt
from defame.utils.console import light_blue
from defame.utils.parsing import extract_first_square_brackets, extract_first_code_block


class ClaimExtractor:
    def __init__(self, llm: Model,
                 prepare_rules: str = '',
                 interpret: bool = False,
                 decompose: bool = False,
                 decontextualize: bool = False,
                 filter_check_worthy: bool = False):
        self.llm = llm
        self.prepare_rules = prepare_rules
        self.do_interpretation = interpret
        self.do_decomposition = decompose
        self.do_decontextualization = decontextualize
        self.do_filtering = filter_check_worthy

        if self.do_decomposition:
            # Requires `python -m spacy download en_core_web_sm`
            from third_party.factscore.atomic_facts import AtomicFactGenerator
            self.atomic_fact_generator = AtomicFactGenerator(
                api_key='', gpt3_cache_file='', other_lm=self.llm
            )

        self.max_retries = 3

    def extract_claims(self, content: Content) -> list[Claim]:
        logger.log(f"Extracting claims from {content}")

        if self.do_interpretation:
            logger.log("Interpreting...")
            self.interpret(content, self.prepare_rules)

        if self.do_decomposition:
            logger.log("Decomposing...")
            claims = self.decompose(content)
            for atomic_fact in claims:
                logger.log(light_blue(f"'{atomic_fact}'"))
        else:
            claims = [Claim(content.text, original_context=content)]

        if self.do_decontextualization:
            logger.log("Decontextualizing...")
            for claim in claims:
                self.decontextualize(claim)
                logger.log(light_blue(f"'{claim}'"))

        if self.do_filtering:
            logger.log("Filtering for unique, check-worthy claims...")
            claims = [claim for claim in claims if self.is_check_worthy(claim)]

        for claim in claims:
            logger.log(light_blue(f"'{claim}'"))

        return claims

    def interpret(self, content: Content, prepare_rules: str = "") -> None:
        """Adds an interpretation to the content object."""
        prompt = InterpretPrompt(content, prepare_rules)
        content.interpretation = self.llm.generate(prompt)

    def decompose(self, content: Content):
        """Splits up the content into atomic facts."""
        interpretation = content.interpretation
        result, _ = self.atomic_fact_generator.run(interpretation)
        atomic_facts = [fact for _, facts in result for fact in facts]
        return atomic_facts

    def decontextualize(self, claim: Claim):
        """Modify the atomic fact to be self-contained."""
        decontextualize_prompt = DecontextualizePrompt(claim)

        model_response, revised_fact, num_tries = '', '', 0
        while not revised_fact and num_tries <= self.max_retries:
            model_response = self.llm.generate(str(decontextualize_prompt))
            revised_fact = extract_first_code_block(
                model_response, ignore_language=True
            )
            num_tries += 1

    def is_check_worthy(self, claim: Claim) -> bool:
        """Identifies whether the given atomic fact is check-worthy."""
        # TODO: adjust the check-worthiness check & analyze the filter behavior...
        # also make sure multiple claims are not almost the same.

        filter_prompt = FilterCheckWorthyPrompt(claim)

        model_response, answer, num_tries = '', '', 0
        while not answer and num_tries <= self.max_retries:
            model_response = self.llm.generate(str(filter_prompt))
            answer = extract_first_square_brackets(model_response)
            answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
            num_tries += 1

        answer = not answer or answer.lower() == SYMBOL.lower()
        return answer
