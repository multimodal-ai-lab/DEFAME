from typing import Any

from infact.common import FCDocument, Label
from infact.common.action import WebSearch
from infact.procedure.procedure import Procedure
from infact.procedure.variants.qa_based.base import extract_queries
from infact.prompts.prompt import ProposeQueriesNoQuestions


class StaticSummary(Procedure):
    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        """InFact but omitting posing any questions."""
        # Stage 2*: Search query generation (modified)
        queries = self.generate_search_queries(doc)

        # Stage 3*: Evidence retrieval (modified)
        results = self.retrieve_resources(queries, summarize=True, doc=doc)
        doc.add_reasoning("## Web Search")
        for result in results[:10]:
            if result.is_useful():
                summary_str = f"### Search Result\n{result}"
                doc.add_reasoning(summary_str)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, {}

    def generate_search_queries(self, doc: FCDocument) -> list[WebSearch]:
        prompt = ProposeQueriesNoQuestions(doc)

        n_tries = 0
        while True:
            n_tries += 1
            response = self.llm.generate(prompt)
            queries = extract_queries(response)

            if len(queries) > 0 or n_tries == self.max_attempts:
                return queries

            self.logger.log("WARNING: No new actions were found. Retrying...")
