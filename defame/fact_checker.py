import multiprocessing
import sys
import time
from typing import Sequence, Any

import numpy as np

from defame.common import logger, Claim, Content, Report, Label, Medium, Action
from defame.common.label import DEFAULT_LABEL_DEFINITIONS
from defame.common.modeling import make_model
from defame.modules.actor import Actor
from defame.modules.claim_extractor import ClaimExtractor
from defame.modules.doc_summarizer import DocSummarizer
from defame.modules.judge import Judge
from defame.modules.planner import Planner
from defame.procedure import get_procedure
from defame.tools import *
from defame.tools.tool import get_available_actions
from defame.utils.console import gray, light_blue, bold, sec2mmss
from defame.utils.utils import cleanup_memory


class FactChecker:
    """The core class for end-to-end fact verification."""

    default_procedure = "defame"

    def __init__(self,
                 llm: str | Model = "gpt_4o_mini",
                 tools: list[Tool] = None,
                 available_actions: list[Action] = None,
                 search_engines: dict[str, dict] = None,
                 procedure_variant: str = None,
                 interpret: bool = False,
                 decompose: bool = False,
                 decontextualize: bool = False,
                 filter_check_worthy: bool = False,
                 max_iterations: int = 5,
                 max_result_len: int = None,
                 restrict_results_to_claim_date: bool = True,
                 classes: Sequence[Label] = None,
                 class_definitions: dict[Label, str] = None,
                 extra_prepare_rules: str = None,
                 extra_plan_rules: str = None,
                 extra_judge_rules: str = None):
        assert not tools or not search_engines, \
            "You are allowed to specify either tools or search engines."

        self.llm = make_model(llm) if isinstance(llm, str) else llm

        self.claim_extractor = ClaimExtractor(llm=self.llm,
                                              prepare_rules=extra_prepare_rules,
                                              interpret=interpret,
                                              decompose=decompose,
                                              decontextualize=decontextualize,
                                              filter_check_worthy=filter_check_worthy)

        if classes is None:
            if class_definitions is None:
                classes = [Label.SUPPORTED, Label.NEI, Label.REFUTED]
                class_definitions = DEFAULT_LABEL_DEFINITIONS
            else:
                classes = list(class_definitions.keys())

        self.extra_prepare_rules = extra_prepare_rules
        self.max_iterations = max_iterations
        self.max_result_len = max_result_len
        self.restrict_results_to_claim_date = restrict_results_to_claim_date

        if tools is None:
            tools = self._initialize_tools(search_engines)

        available_actions = get_available_actions(tools, available_actions)

        # Initialize fact-checker modules
        self.planner = Planner(valid_actions=available_actions,
                               llm=self.llm,
                               extra_rules=extra_plan_rules)

        self.actor = Actor(tools=tools)

        self.judge = Judge(llm=self.llm,
                           classes=classes,
                           class_definitions=class_definitions,
                           extra_rules=extra_judge_rules)

        self.doc_summarizer = DocSummarizer(self.llm)

        if procedure_variant is None:
            procedure_variant = self.default_procedure

        self.procedure = get_procedure(procedure_variant,
                                       llm=self.llm,
                                       actor=self.actor,
                                       judge=self.judge,
                                       planner=self.planner,
                                       max_iterations=self.max_iterations)

    def _initialize_tools(self, search_engines: dict[str, dict]) -> list[Tool]:
        """Loads a default collection of tools."""
        tools = [
            Searcher(search_engines, max_result_len=self.max_result_len, llm=self.llm),
            # CredibilityChecker(),
            Geolocator(top_k=10),
            # ManipulationDetector(),
        ]

        return tools

    def check_content(self, content: Content | str) -> tuple[Label, list[Report], list[dict[str, Any]]]:
        """
        Fact-checks the given content ent-to-end by first extracting all check-worthy claims and then
        verifying each claim individually. Returns the aggregated veracity and the list of corresponding
        fact-checking documents, one doc per claim.
        """
        start = time.time()

        content = Content(content) if isinstance(content, str) else content

        claims = self.claim_extractor.extract_claims(content)

        # Verify each single extracted claim
        docs = []
        metas = []
        for claim in claims:  # TODO: parallelize
            doc, meta = self.verify_claim(claim)
            docs.append(doc)
            metas.append(meta)
            doc.save_to(logger.target_dir)

        aggregated_veracity = aggregate_predictions([doc.verdict for doc in docs])
        logger.log(bold(f"So, the overall veracity is: {aggregated_veracity.value}"))
        fc_duration = time.time() - start
        logger.log(f"Fact-check took {sec2mmss(fc_duration)}.")
        return aggregated_veracity, docs, metas

    def verify_claim(self, claim: Claim | list[str | Medium]) -> tuple[Report, dict[str, Any]]:
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        if isinstance(claim, list):
            claim = Claim(claim)

        logger.info(f"Verifying {bold(str(claim))}")

        stats = {}
        self.actor.reset()  # remove all past search evidences
        if self.restrict_results_to_claim_date:
            self.actor.set_search_date_restriction(claim.date)
        if not self.llm:
            worker_name = multiprocessing.current_process().name
            logger.critical(f"No LLM was loaded. Stopping execution for {worker_name}.")
            sys.exit(1)  # Exits the process for this worker
        self.llm.reset_stats()

        start = time.time()
        doc = Report(claim)

        # Depending on the specified procedure variant, perform the fact-check
        label, meta = self.procedure.apply_to(doc)

        # Finalize the fact-check
        doc.add_reasoning("## Final Judgement\n" + self.judge.get_latest_reasoning())

        # Summarize the fact-check and use the summary as justification
        if label == Label.REFUSED_TO_ANSWER:
            logger.warning("The model refused to answer.")
        else:
            doc.justification = self.doc_summarizer.summarize(doc)
            logger.info(bold(f"The claim '{light_blue(str(claim.text))}' is {label.value}."))
            logger.info(f'Justification: {gray(doc.justification)}')
        doc.verdict = label

        stats["Duration"] = time.time() - start
        stats["Model"] = self.llm.get_stats()
        stats["Tools"] = self.actor.get_tool_stats()
        meta["Statistics"] = stats

        cleanup_memory()

        return doc, meta


def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    # If all predicted labels are the same label, return that label
    if len(set(veracities)) == 1:
        return veracities[0]

    # Otherwise, apply this aggregation
    veracities = np.array(veracities)
    if np.any(veracities == Label.REFUSED_TO_ANSWER):
        return Label.REFUSED_TO_ANSWER
    elif np.any(veracities == Label.REFUTED):
        return Label.REFUTED
    elif np.any(veracities == Label.CONFLICTING):
        return Label.CONFLICTING
    elif np.any(veracities == Label.CHERRY_PICKING):
        return Label.CHERRY_PICKING
    else:
        return Label.NEI
