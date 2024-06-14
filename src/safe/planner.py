from common.action import Action, WebSearch, WikiLookup, ACTION_REGISTRY
from common.console import orange
from common.document import FCDocument
from common.modeling import Model
from common.utils import extract_last_code_block, remove_code_blocks
from eval.logger import EvaluationLogger
from safe.prompts.prompt import PlanPrompt
from typing import Optional
import pyparsing as pp


class Planner:
    """Takes a fact-checking document and proposes the next actions to take
    based on the current knowledge as contained in the document."""

    def __init__(self, valid_actions: list[type[Action]],
                 model: Model,
                 logger: EvaluationLogger,
                 extra_rules: str):
        assert len(valid_actions) > 0
        self.valid_actions = valid_actions
        self.model = model
        self.logger = logger
        self.max_tries = 5
        self.extra_rules = extra_rules

    def plan_next_actions(self, doc: FCDocument) -> (list[Action], str):
        prompt = PlanPrompt(doc, self.valid_actions, self.extra_rules)
        n_tries = 0
        while True:
            n_tries += 1
            answer = self.model.generate(str(prompt))
            actions = self._extract_actions(answer)
            reasoning = self._extract_reasoning(answer)
            if len(actions) > 0 or n_tries == self.max_tries:
                return actions, reasoning
            self.logger.log("WARNING: No actions were found. Retrying...")

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

    def _extract_reasoning(self, answer: str) -> str:
        return remove_code_blocks(answer).strip()

    def _parse_single_action(self, raw_action: str) -> Optional[Action]:
        try:
            action_name, arguments_str = raw_action.split(':', maxsplit=1)
            arguments = _extract_arguments(arguments_str)
            for action in ACTION_REGISTRY:
                if action_name == action.name:
                    return action(*arguments)
            raise ValueError(f'Invalid action name: {action_name}')
        except Exception as e:
            self.logger.log(orange(f"WARNING: Failed to parse '{raw_action}':\n{e}"))
        return None


def _process_answer(answer: str) -> str:
    reasoning = answer.split("NEXT_ACTIONS:")[0].strip()
    return reasoning.replace("REASONING:", "").strip()


def _extract_arguments(arguments_str: str) -> list[str]:
    """Separates the arguments_str at all commas that are not enclosed by quotes."""
    ppc = pp.pyparsing_common

    # Setup parser which separates at each comma not enclosed by a quote
    csl = ppc.comma_separated_list()

    # Parse the string using the created parser
    parsed = csl.parse_string(arguments_str)

    # Remove whitespaces and split into arguments list
    return [str.strip(value) for value in parsed]
