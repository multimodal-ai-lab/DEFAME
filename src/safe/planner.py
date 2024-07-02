from common.action import Action, WebSearch, WikiDumpLookup, WikiLookup, ACTION_REGISTRY
from common.console import orange
from common.document import FCDocument
from common.modeling import Model
from common.utils import extract_last_code_block, remove_code_blocks
from eval.logger import EvaluationLogger
from safe.prompts.prompt import PlanPrompt
from typing import Optional, Tuple
import pyparsing as pp
import re
from PIL import Image


class Planner:
    """Takes a fact-checking document and proposes the next actions to take
    based on the current knowledge as contained in the document."""

    def __init__(self, 
                 multimodal: bool,
                 valid_actions: list[type[Action]],
                 model: Model,
                 logger: EvaluationLogger,
                 extra_rules: str):
        assert len(valid_actions) > 0
        self.multimodal = multimodal
        self.valid_actions = valid_actions
        self.model = model
        self.logger = logger
        self.max_tries = 5
        self.extra_rules = extra_rules

    def plan_next_actions(self, doc: FCDocument, images: Optional[list[Image.Image]] = None) -> (list[Action], str):
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
            answer = self.model.generate(str(prompt))
            actions = self._extract_actions(answer, images)
            reasoning = self._extract_reasoning(answer)

            # Filter out actions that have been performed before
            actions = [action for action in actions if action not in performed_actions]

            if len(actions) > 0 or n_tries == self.max_tries:
                return actions, reasoning

            self.logger.log("WARNING: No new actions were found. Retrying...")



    def _extract_actions(self, answer: str, images: Optional[list[Image.Image]] = None) -> list[Action]:
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
            action = self._parse_single_action(raw_action, images)
            if action:
                actions.append(action)
        return actions

    def _extract_reasoning(self, answer: str) -> str:
        return remove_code_blocks(answer).strip()

    def _parse_single_action(self, raw_action: str, images: Optional[list[Image.Image]] = None, fallback = 'wiki_dump_lookup') -> Optional[Action]:
        try:
            # Use regular expression to match action and argument in the form action(argument)
            match = re.match(r'(\w+)\((.*)\)', raw_action)
            if match:
                action_name, argument = match.groups()
                argument = argument.strip()
            else:
                self.logger.log(f"Invalid action format: {raw_action}")
                match = re.search(r'"(.*?)"', raw_action)
                argument = f'"{match.group(1)}"' if match else f'"{raw_action}"'
                first_part = raw_action.split(' ')[0]
                action_name = re.sub(r'[^a-zA-Z0-9\_]', '', first_part)
            if "image" in argument:
                #TODO: implement multi image argument
                argument = images[0]
            for action in ACTION_REGISTRY:
                if action_name == action.name:
                    return action(argument) 
            raise ValueError(f'Invalid action format. Fallback to {fallback} with argument: {argument}')
        except Exception as e:
            self.logger.log(f"WARNING: Failed to parse '{raw_action}':\n{e}")
            fallback_action = next((action for action in ACTION_REGISTRY if fallback == action.name), None)
            if not isinstance(argument, str):
                return None
            elif not(argument[0] == argument[-1] == '"'):
                argument = f'"{argument}"'
        return fallback_action(argument)


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

