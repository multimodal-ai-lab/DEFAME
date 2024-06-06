from common.modeling import Model
from common.document import FCDoc
from eval.logger import EvaluationLogger
from common.action import Action, SearchAction
from safe.prompts.prompt import PlanPrompt
from common.utils import extract_last_code_block


class Planner:
    """Takes a fact-checking document and proposes the next actions to take
    based on the current knowledge as contained in the document."""

    def __init__(self, model: Model, logger: EvaluationLogger):
        self.model = model
        self.logger = logger

    def plan_next_actions(self, doc: FCDoc) -> (list[Action], str):
        prompt = PlanPrompt(doc)
        answer = self.model.generate(str(prompt))
        reasoning = answer.split("NEXT_ACTIONS:")[0].strip()
        return _extract_actions(answer), reasoning


def _extract_actions(answer: str) -> list[Action]:
    actions_str = extract_last_code_block(answer)
    if not actions_str:
        return []
    raw_actions = actions_str.split('\n')
    actions = []
    for raw_action in raw_actions:
        action_name, arguments_str = raw_action.split(':')
        arguments = [a.strip() for a in arguments_str.split(',')]
        action = SearchAction(action_name, arguments[0])
        actions.append(action)
    return actions
