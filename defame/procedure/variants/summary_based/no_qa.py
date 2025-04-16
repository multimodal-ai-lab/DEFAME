from typing import Any

from defame.common import Report, Label, logger
from defame.evidence_retrieval.tools import Search
from defame.procedure.procedure import Procedure
from defame.prompts.prompts import ProposeQueriesNoQuestions


class NoQA(Procedure):
    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        """InFact but omitting posing any questions."""
        # Stage 2*: Search query generation (modified)
        queries = self.generate_search_queries(doc)

        # Stage 3*: Evidence retrieval (modified)
        results = self.retrieve_sources(queries, summarize=True, doc=doc)
        doc.add_reasoning("## Web Search")
        for result in results[:10]:
            if result.is_relevant():
                summary_str = f"### Search Result\n{result}"
                doc.add_reasoning(summary_str)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, {}

    def generate_search_queries(self, doc: Report) -> list[Search]:
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

            logger.log("No new actions were found. Retrying...")

        logger.warning("Got no search query, dropping this question.")
        return []
