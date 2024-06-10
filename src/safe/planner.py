from common.action import Action, WebSearch, WikiLookup
from common.console import orange
from common.document import FCDocument
from common.modeling import Model
from common.utils import extract_last_code_block
from eval.logger import EvaluationLogger
from safe.prompts.prompt import PlanPrompt
from typing import Optional


class Planner:
    """Takes a fact-checking document and proposes the next actions to take
    based on the current knowledge as contained in the document."""

    def __init__(self, valid_actions: list[type[Action]], model: Model, logger: EvaluationLogger):
        assert len(valid_actions) > 0
        self.valid_actions = valid_actions
        self.model = model
        self.logger = logger

    def plan_next_actions(self, doc: FCDocument) -> (list[Action], str):
        prompt = PlanPrompt(doc, self.valid_actions)
        answer = self.model.generate(str(prompt))
        reasoning = _process_answer(answer)
        return self._extract_actions(answer), reasoning

    def _extract_actions(self, answer: str) -> list[Action]:
        actions_str = extract_last_code_block(answer)
        if not actions_str:
            return []
        raw_actions = actions_str.split('\n')
        actions = []
        for raw_action in raw_actions:
            action = self._parse_single_action(raw_action)
            if action:
                actions.append(action)
        return actions

    def _parse_single_action(self, raw_action: str) -> Optional[Action]:
        try:
            action_name, arguments_str = raw_action.split(':')
            arguments = [a.strip() for a in arguments_str.split(',')]
            match action_name:
                case "WEB_SEARCH":
                    return WebSearch(arguments[0])
                case "WIKI_LOOKUP":
                    return WikiLookup(arguments[0])
                case _:
                    self.logger.log(orange(f"WARNING: Unrecognized action '{action_name}'"))
        except Exception:
            self.logger.log(orange(f"WARNING: Failed to parse '{raw_action}'."))
        return None


def _process_answer(answer: str) -> str:
    reasoning = answer.split("NEXT_ACTIONS:")[0].strip()
    return reasoning.replace("REASONING:", "").strip()
