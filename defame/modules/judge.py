import dataclasses

from defame.common import Report, logger, Model, Prompt, Label
from defame.common.label import DEFAULT_LABEL_DEFINITIONS
from defame.prompts.prompts import JudgePrompt, JudgeNaively, JudgeMinimal


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class Judge:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self,
                 llm: Model,
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

    def judge(self, doc: Report, is_final: bool = True) -> Label:
        classes = self.classes.copy()

        # If this is a non-final judgement (i.e. there are follow-up retrievals/actions allowed)
        # enable to predict NEI (otherwise fact-check would always end here)
        if not is_final:
            classes.add(Label.NEI)

        prompt = JudgePrompt(doc, classes, self.class_definitions, self.extra_rules)
        return self._generate_verdict(prompt)

    def judge_naively(self, doc: Report) -> Label:
        prompt = JudgeNaively(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt)

    def judge_minimally(self, doc: Report) -> Label:
        prompt = JudgeMinimal(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt)

    def _generate_verdict(self, prompt: Prompt) -> Label:
        response = self.llm.generate(prompt)

        if response is None or not response["verdict"]:
            logger.warning(f"Error while generating verdict. Defaulting to REFUSED.")
            self.latest_reasoning = ""
            return Label.REFUSED_TO_ANSWER

        self.latest_reasoning = response["response"]
        return response["verdict"]

    def get_latest_reasoning(self) -> str:
        return self.latest_reasoning
