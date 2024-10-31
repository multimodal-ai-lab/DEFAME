import dataclasses

from infact.common.document import FCDocument
from infact.common.label import Label, DEFAULT_LABEL_DEFINITIONS
from infact.common.logger import Logger
from infact.common.modeling import Model
from infact.common.prompt import Prompt
from infact.prompts.prompts import JudgePrompt, JudgeNaively
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
        self.classes = set(classes)

        if Label.NEI not in class_definitions:
            class_definitions[Label.NEI] = DEFAULT_LABEL_DEFINITIONS[Label.NEI]
        self.class_definitions = class_definitions

        self.extra_rules = extra_rules
        self.max_retries = 5
        self.latest_reasoning = None

        self.logger = logger

    def judge(self, doc: FCDocument, is_final: bool = True) -> Label:
        classes = self.classes.copy()

        # If this is a non-final judgement (i.e. there are follow-up retrievals/actions allowed)
        # enable to predict NEI (otherwise fact-check would always end here)
        if not is_final:
            classes.add(Label.NEI)

        prompt = JudgePrompt(doc, classes, self.class_definitions, self.extra_rules)
        return self._generate_verdict(prompt)

    def judge_naively(self, doc: FCDocument) -> Label:
        prompt = JudgeNaively(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt)

    def _generate_verdict(self, prompt: Prompt) -> Label:
        response = self.llm.generate(prompt)

        if not response["verdict"]:
            self.logger.info(orange(f"Error while generating verdict for response: {response['response']}\n\nDefaulting to REFUSED."))
            self.latest_reasoning = ""
            return Label.REFUSED_TO_ANSWER

        self.latest_reasoning = response["response"]
        return response["verdict"]

    def get_latest_reasoning(self) -> str:
        return self.latest_reasoning
