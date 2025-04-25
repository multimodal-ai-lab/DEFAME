import multiprocessing
import sys
import time
from typing import Sequence, Any
from datetime import datetime

import numpy as np
from ezmm import Item

from defame.common import logger, Claim, Content, Report, Label, Action, Model
from defame.common.label import DEFAULT_LABEL_DEFINITIONS
from defame.common.modeling import make_model
from defame.modules.actor import Actor
from defame.modules.claim_extractor import ClaimExtractor
from defame.modules.doc_summarizer import DocSummarizer
from defame.modules.judge import Judge
from defame.modules.planner import Planner
from defame.procedure import get_procedure
from defame.evidence_retrieval import scraper, Tool
from defame.evidence_retrieval.tools import initialize_tools
from defame.evidence_retrieval.tools.tool import get_available_actions
from defame.utils.console import gray, light_blue, bold, sec2mmss


class FactChecker:
    """The core class for end-to-end fact verification."""

    default_procedure = "defame"

    def __init__(self,
                 llm: str | Model = "gpt_4o_mini",
                 llm_kwargs: dict = None,
                 tools: list[Tool] = None,
                 tools_config: dict = None,
                 available_actions: list[Action] = None,
                 procedure_variant: str = None,
                 interpret: bool = False,
                 decompose: bool = False,
                 decontextualize: bool = False,
                 filter_check_worthy: bool = False,
                 max_iterations: int = 5,
                 max_result_len: int = None,
                 restrict_results_to_claim_date: bool = True,
                 allow_fact_checking_sites: bool = True,
                 classes: Sequence[Label] = None,
                 class_definitions: dict[Label, str] = None,
                 extra_prepare_rules: str = None,
                 extra_plan_rules: str = None,
                 extra_judge_rules: str = None,
                 device: str = None):

        if tools_config is None:
            tools_config = dict(searcher=None)

        if llm_kwargs is None:
            llm_kwargs = {}

        if device is not None:
            llm_kwargs.update(device=device)

        self.llm = make_model(llm, **llm_kwargs) if isinstance(llm, str) else llm

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
        scraper.allow_fact_checking_sites = allow_fact_checking_sites

        if tools is None:
            tools = initialize_tools(tools_config, llm=self.llm)

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

    def extract_claims(self, content: Content | list[str | Item]) -> list[Claim]:
        if not isinstance(content, Content):
            content = Content(content)
        return self.claim_extractor.extract_claims(content)

    def check_content(self, content: Content | list[str | Item]) -> tuple[Label, list[Report], list[dict[str, Any]]]:
        """
        Fact-checks the given content ent-to-end by first extracting all check-worthy claims and then
        verifying each claim individually. Returns the aggregated veracity and the list of corresponding
        fact-checking documents, one doc per claim.
        """
        start = time.time()

        claims = self.extract_claims(content)

        # Verify each single extracted claim
        docs = []
        metas = []
        for claim in claims:  # TODO: parallelize
            doc, meta = self.verify_claim(claim)
            docs.append(doc)
            metas.append(meta)
            target_dir = logger.target_dir if logger.target_dir else "out/fact_check"
            doc.save_to(target_dir)

        aggregated_veracity = aggregate_predictions([doc.verdict for doc in docs])
        logger.log(bold(f"So, the overall veracity is: {aggregated_veracity.value}"))
        fc_duration = time.time() - start
        logger.log(f"Fact-check took {sec2mmss(fc_duration)}.")
        return aggregated_veracity, docs, metas

    def verify_claim(self, claim: Claim | list[str | Item]) -> tuple[Report, dict[str, Any]]:
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        if not isinstance(claim, Claim):
            claim = Claim(claim)

        logger.info(f"Verifying claim.", send=True)
        logger.info(f"{bold(str(claim))}")

        stats = {}
        self.actor.reset()  # remove all past search evidences
        self.actor.set_current_claim_id(claim.id)
        if self.restrict_results_to_claim_date:
            # Set the restriction to midnight of the claim date
            restriction_time = datetime.combine(claim.date, datetime.min.time()) if claim.date else None
            self.actor.set_search_date_restriction(restriction_time)
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
            logger.info(bold(f"The claim '{light_blue(str(claim))}' is {label.value}."))
            logger.info(f'Justification: {gray(doc.justification)}')
        doc.verdict = label

        stats["Duration"] = time.time() - start
        stats["Model"] = self.llm.get_stats()
        stats["Tools"] = self.actor.get_tool_stats()
        meta["Statistics"] = stats
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
