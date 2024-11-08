from typing import Any

from infact.common import FCDocument, Label
from infact.tools import WebSearch
from infact.procedure.procedure import Procedure
from infact.prompts.prompts import ProposeQueriesNoQuestions


class NoQA(Procedure):
    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        """InFact but omitting posing any questions."""
        # Stage 2*: Search query generation (modified)
        queries = self.generate_search_queries(doc)

        # Stage 3*: Evidence retrieval (modified)
        results = self.retrieve_resources(queries, summarize=True, doc=doc)
        doc.add_reasoning("## Web Search")
        for result in results[:10]:
            if result.is_relevant():
                summary_str = f"### Search Result\n{result}"
                doc.add_reasoning(summary_str)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, {}

    def generate_search_queries(self, doc: FCDocument) -> list[WebSearch]:
        prompt = ProposeQueriesNoQuestions(doc)

        n_attempts = 0
        while n_attempts < self.max_attempts:
            n_attempts += 1
            response = self.llm.generate(prompt)

            if response is None:
                continue

            queries: list = response["queries"]

            if len(queries) > 0:
                return queries

            self.logger.log("No new actions were found. Retrying...")

        self.logger.warning("Got no search query, dropping this question.")
        return []
