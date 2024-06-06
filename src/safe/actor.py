from common.action import Action, SearchAction
from common.modeling import Model
from common.results import Result
from eval.logger import EvaluationLogger
from safe.searcher import Searcher


class Actor:
    def __init__(self,
                 model: str | Model = "OPENAI:gpt-3.5-turbo-0125",
                 search_engines: list[str] = None,
                 max_searches_per_claim: int = 5,
                 logger: EvaluationLogger = None,
                 ):

        self.logger = logger or EvaluationLogger()

        search_engines = search_engines or ["duckduck"]
        self.searcher = Searcher(search_engines, model, self.logger,
                                 summarize=False,
                                 limit_per_search=max_searches_per_claim)

    def perform(self, actions: list[Action]) -> list[Result]:
        # TODO: Enable parallelization here, e.g. through async calls
        all_results = []
        for a in actions:
            all_results.extend(self._perform_single(a))
        return all_results

    def _perform_single(self, action: Action) -> list[Result]:
        if isinstance(action, SearchAction):
            return self._perform_search(action)
        else:
            raise ValueError(f"Action '{action}' unknown.")

    def _perform_search(self, search_action: SearchAction) -> list[Result]:
        match search_action.api:
            case "WEB_SEARCH": return self.searcher.search(search_action.query)
            case "WIKI_LOOKUP": return self.searcher.search(search_action.query)  # TODO
            case _: raise ValueError(f"Unknown search API '{search_action.api}'.")
