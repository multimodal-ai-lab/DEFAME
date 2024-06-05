from common.modeling import Model
from common.document import FCDoc
from eval.logger import EvaluationLogger
from safe.actor import Action


class Planner:
    """Takes a fact-checking document and proposes the next actions to take."""

    def __init__(self, model: Model, logger: EvaluationLogger):
        self.model = model
        self.logger = logger

    def plan_next_actions(self, doc: FCDoc) -> list[Action]:
        pass
