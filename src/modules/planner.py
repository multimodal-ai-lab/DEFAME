import re
from typing import Optional, Collection

import pyparsing as pp
from PIL import Image

from src.common.action import (Action, ACTION_REGISTRY, IMAGE_ACTIONS)
from src.common.content import Content
from src.common.document import FCDocument
from src.common.modeling import LLM
from src.eval.logger import EvaluationLogger
from src.prompts.prompt import PlanPrompt
from src.utils.parsing import extract_last_code_block, remove_code_blocks


class Planner:
    """Chooses the next actions to perform based on the current knowledge as contained
    in the FC document."""

    def __init__(self,
                 valid_actions: Collection[type[Action]],
                 llm: LLM,
                 logger: EvaluationLogger,
                 extra_rules: str):
        assert len(valid_actions) > 0
        self.valid_actions = valid_actions
        self.llm = llm
        self.logger = logger
        self.max_tries = 5
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
        n_tries = 0

        while True:
            n_tries += 1
            answer = self.llm.generate(str(prompt))
            actions = self._extract_actions(answer, doc.claim.original_context)
            reasoning = self._extract_reasoning(answer)

            # Filter out actions that have been performed before
            actions = [action for action in actions if action not in performed_actions]

            if len(actions) > 0 or n_tries == self.max_tries:
                return actions, reasoning

            self.logger.log("WARNING: No new actions were found. Retrying...")

    def _extract_actions(self, answer: str, context: Content = None) -> list[Action]:
        actions_str = extract_last_code_block(answer)
        if not actions_str:
            candidates = []
            for action in ACTION_REGISTRY:
                pattern = re.compile(f'{action.name}("(.*?)")', re.DOTALL)
                candidates += pattern.findall(answer)
            actions_str = "\n".join(candidates)
        if not actions_str:
            return []
        raw_actions = actions_str.split('\n')
        actions = []
        for raw_action in raw_actions:
            action = self._parse_single_action(raw_action, context.images)
            if action:
                actions.append(action)
        return actions

    def _extract_reasoning(self, answer: str) -> str:
        return remove_code_blocks(answer).strip()

    def _parse_single_action(self, raw_action: str, images: Optional[list[Image.Image]] = None,
                             fallback='wiki_dump_lookup') -> Optional[Action]:
        arguments = None

        try:
            # Use regular expression to match action and argument in the form action(argument)
            match = re.match(r'(\w+)\((.*)\)', raw_action)

            # Extract action name and arguments
            if match:
                action_name, arguments = match.groups()
                arguments = arguments.strip()
            else:
                self.logger.log(f"Invalid action format: {raw_action}")
                match = re.search(r'"(.*?)"', raw_action)
                arguments = f'"{match.group(1)}"' if match else f'"{raw_action}"'
                first_part = raw_action.split(' ')[0]
                action_name = re.sub(r'[^a-zA-Z0-9_]', '', first_part)

            if "image" in arguments:
                # TODO: implement multi image argument
                arguments = images[0]

            for action in ACTION_REGISTRY:
                if action_name == action.name:
                    return action(arguments)

            raise ValueError(f'Invalid action format. Fallback to {fallback} with argument: {arguments}')

        except Exception as e:
            self.logger.log(f"WARNING: Failed to parse '{raw_action}':\n{e}")

        # Try fallback parsing
        try:
            fallback_action = next((action for action in ACTION_REGISTRY if fallback == action.name), None)
            if not isinstance(arguments, str):
                return None
            elif not (arguments[0] == arguments[-1] == '"'):
                arguments = f'"{arguments}"'

            return fallback_action(arguments)

        except Exception as e:
            self.logger.log(f"WARNING: Failed to parse '{raw_action}' even during fallback parsing:\n{e}")


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
