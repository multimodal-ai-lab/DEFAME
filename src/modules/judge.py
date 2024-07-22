import dataclasses
import re

from src.common.document import FCDocument
from src.common.label import Label
from src.common.modeling import LLM
from src.eval.logger import EvaluationLogger
from src.prompts.prompt import JudgePrompt
from src.utils.console import orange
from src.utils.parsing import extract_last_code_span, is_guardrail_hit, GUARDRAIL_WARNING


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class Judge:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self,
                 llm: LLM,
                 logger: EvaluationLogger,
                 classes: list[Label],
                 class_definitions: dict[Label, str] = None,
                 extra_rules: str = None,
                 do_debug: bool = False):
        self.llm = llm
        self.classes = classes
        self.class_definitions = class_definitions
        self.extra_rules = extra_rules
        self.debug = do_debug
        self.max_retries = 5
        self.latest_reasoning = None

        self.logger = logger

    def judge(self, doc: FCDocument) -> Label:
        judge_prompt = JudgePrompt(doc, self.classes, self.class_definitions, self.extra_rules)
        n_tries = 0
        while (verdict := self._generate_verdict(str(judge_prompt))) == Label.REFUSED_TO_ANSWER:
            n_tries += 1
            if n_tries > self.max_retries:
                ####################### AVeriTeC Specific #######################
                self.logger.log(orange(f"Model refused to answer. Fallback to REFUTED Label."))
                verdict = Label.REFUTED
                #################################################################
                break
            self.logger.log(orange("WARNING: Verdict generation did not contain any valid label. Retrying..."))
        return verdict

    def _generate_verdict(self, prompt: str) -> Label:
        # Generate an answer
        try:
            response = self.llm.generate(prompt, do_debug=self.debug)
            self.latest_reasoning = response
        except Exception as e:
            self.logger.log(orange("WARNING: Error when generating verdict, defaulting to REFUSED."))
            self.logger.log(orange(str(e)))
            return Label.REFUSED_TO_ANSWER

        # Validate model response
        if is_guardrail_hit(response):
            self.logger.log(GUARDRAIL_WARNING)
            self.logger.log(orange("PROMPT:\n" + prompt))
            return Label.REFUSED_TO_ANSWER

        # Extract the verdict
        verdict = self._extract_verdict(response)
        if verdict == Label.REFUSED_TO_ANSWER:
            self.logger.log(orange(f"WARNING: Ill-formatted verdict in response:\n{response}"))
        return verdict

    def _extract_verdict(self, response: str) -> Label:
        """Extract label from response"""
        answer = extract_last_code_span(response)
        answer = re.sub(r'[^\w\-\s]', '', answer).strip().lower()

        if not answer:
            pattern = re.compile(r'\*\*(.*)\*\*', re.DOTALL)
            matches = pattern.findall(response) or ['']
            answer = matches[0]
        try:
            verdict = Label(answer)
        except ValueError:
            # Maybe the label is a substring of the response
            for c in self.classes:
                if c.value in response:
                    return c

            verdict = Label.REFUSED_TO_ANSWER

        return verdict

    def get_latest_reasoning(self) -> str:
        return self.latest_reasoning
