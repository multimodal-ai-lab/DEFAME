from typing import Collection

import pyparsing as pp

from infact.common.action import (Action)
from infact.tools import IMAGE_ACTIONS
from infact.common.document import FCDocument
from infact.common.modeling import Model
from infact.common.logger import Logger
from infact.prompts.prompts import PlanPrompt


class Planner:
    """Chooses the next actions to perform based on the current knowledge as contained
    in the FC document."""

    def __init__(self,
                 valid_actions: Collection[type[Action]],
                 llm: Model,
                 logger: Logger,
                 extra_rules: str
                 ):
        self.valid_actions = valid_actions
        self.llm = llm
        self.logger = logger
        self.max_attempts = 5
        self.extra_rules = extra_rules

    def get_available_actions(self, doc: FCDocument):
        available_actions = []
        completed_actions = set(type(a) for a in doc.get_all_actions())

        if doc.claim.has_image():  # TODO: enable multiple image actions for multiple images
            available_actions += [a for a in IMAGE_ACTIONS if a not in completed_actions]

        # TODO: finish this method

        return available_actions

    def plan_next_actions(self, doc: FCDocument) -> (list[Action], str):
        # TODO: include image in planning
        performed_actions = doc.get_all_actions()
        new_valid_actions = []

        # Check if actions have been performed before adding them to valid actions
        for action_class in self.valid_actions:
            is_performed = False
            for action in performed_actions:
                if isinstance(action, action_class):
                    is_performed = True
                    break

            if not action_class.is_multimodal or (action_class.is_multimodal and not is_performed):
                new_valid_actions.append(action_class)
            else:
                self.logger.log(f"INFO: Dropping action '{action_class.name}' as it was already performed.")

        self.valid_actions = new_valid_actions
        prompt = PlanPrompt(doc, self.valid_actions, self.extra_rules)
        n_attempts = 0

        while n_attempts < self.max_attempts:
            n_attempts += 1

            response = self.llm.generate(prompt)
            if response is None:
                self.logger.warning("No new actions were found.")
                return [], ""


            actions = response["actions"]
            reasoning = response["reasoning"]

            # Filter out actions that have been performed before
            actions = [action for action in actions if action not in performed_actions]

            if len(actions) > 0:
                return actions, reasoning
            else:
                self.logger.warning(f'No actions were found in this response: {response["response"]}')
                return [], ""


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
