from infact.common.action import Action
from infact.common.document import FCDocument
from infact.common.modeling import Model
from infact.common.results import Evidence
from infact.common.logger import Logger
from infact.modules.result_summarizer import ResultSummarizer
from infact.tools.tool import Tool


class Actor:
    """Agent that executes given Actions and returns the resulted Evidence."""

    def __init__(self, tools: list[Tool], llm: Model, logger: Logger):
        self.tools = tools
        self.result_summarizer = ResultSummarizer(llm, logger)

    def perform(self, actions: list[Action], doc: FCDocument = None, summarize: bool = True) -> list[Evidence]:
        # TODO: Parallelize
        all_evidence = []
        for action in actions:
            all_evidence.append(self._perform_single(action, doc, summarize=summarize))
        return all_evidence

    def _perform_single(self, action: Action, doc: FCDocument = None, summarize: bool = True) -> Evidence:
        tool = self.get_corresponding_tool_for_action(action)
        results = tool.perform(action)
        if summarize:
            assert doc is not None
            results = self.result_summarizer.summarize(results, doc)
        summary = ""  # TODO: Summarize result summaries
        return Evidence(summary, list(results))

    def get_corresponding_tool_for_action(self, action: Action) -> Tool:
        for tool in self.tools:
            if type(action) in tool.actions:
                return tool
        raise ValueError(f"No corresponding tool available for Action '{action}'.")

    def reset(self):
        """Resets all tools (if applicable)."""
        for tool in self.tools:
            tool.reset()
    