import dataclasses

from infact.common.document import FCDocument
from infact.common.label import Label
from infact.common.logger import Logger
from infact.common.modeling import Model
from infact.prompts.prompt import JudgePrompt, JudgeNaively, Prompt
from infact.utils.console import orange


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class Judge:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self,
                 llm: Model,
                 logger: Logger,
                 classes: list[Label],
                 class_definitions: dict[Label, str] = None,
                 extra_rules: str = None):
        self.llm = llm
        self.classes = classes
        self.class_definitions = class_definitions
        self.extra_rules = extra_rules
        self.max_retries = 5
        self.latest_reasoning = None

        self.logger = logger

    def judge(self, doc: FCDocument) -> Label:
        prompt = JudgePrompt(doc, self.classes, self.class_definitions, self.extra_rules)
        return self._generate_verdict(prompt)

    def judge_naively(self, doc: FCDocument) -> Label:
        prompt = JudgeNaively(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt)

    def _generate_verdict(self, prompt: Prompt) -> Label:
        response = self.llm.generate(prompt)

        if response is None:
            self.logger.log(orange("Error while generating verdict, defaulting to REFUSED."))
            self.latest_reasoning = None
            return Label.REFUSED_TO_ANSWER

        self.latest_reasoning = response["response"]
        return response["verdict"]

    def get_latest_reasoning(self) -> str:
        return self.latest_reasoning
