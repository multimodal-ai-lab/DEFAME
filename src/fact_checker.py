from typing import Sequence, Optional, Collection

from src.common.action import *
from src.common.claim import Claim
from src.common.content import Content
from src.common.document import FCDocument, ReasoningBlock
from src.common.label import Label
from src.common.modeling import LLM, MLLM
from src.common.results import Evidence
from src.modules.actor import Actor
from src.modules.claim_extractor import ClaimExtractor
from src.modules.doc_summarizer import DocSummarizer
from src.modules.judge import Judge
from src.modules.planner import Planner
from src.prompts.prompt import PoseQuestionsPrompt, ReiteratePrompt
from src.tools import *
from src.utils.console import gray, light_blue, bold


class FactChecker:
    """The core class for end-to-end fact verification."""

    def __init__(self,
                 llm: str | LLM = "OPENAI:gpt-3.5-turbo-0125",
                 mllm: Optional[str | MLLM] = None,
                 tools: list[Tool] = None,
                 search_engines: list[str] = None,
                 interpret: bool = False,
                 decompose: bool = False,
                 decontextualize: bool = False,
                 filter_check_worthy: bool = False,
                 max_iterations: int = 5,
                 logger: EvaluationLogger = None,
                 classes: Sequence[Label] = None,
                 class_definitions: dict[Label, str] = None,
                 extra_prepare_rules: str = None,
                 extra_plan_rules: str = None,
                 extra_judge_rules: str = None,
                 verbose: bool = True,
                 ):
        assert tools is None or search_engines is None, \
            "You are allowed to specify either tools or search engines."

        self.logger = logger or EvaluationLogger(verbose=verbose)

        self.llm = LLM(llm, self.logger) if isinstance(llm, str) else llm
        self.mllm = MLLM(mllm, self.logger) if (isinstance(mllm, str)) else mllm

        self.claim_extractor = ClaimExtractor(llm=self.llm,
                                              mllm=self.mllm,
                                              interpret=interpret,
                                              decompose=decompose,
                                              decontextualize=decontextualize,
                                              filter_check_worthy=filter_check_worthy,
                                              logger=self.logger)

        if classes is None:
            if class_definitions is None:
                classes = [Label.SUPPORTED, Label.NEI, Label.REFUTED]
            else:
                classes = list(class_definitions.keys())

        if tools is None:
            tools = self._initialize_tools(search_engines)
        self.fall_back_action = tools[0].actions[0]
        logger.log(f"Selecting {self.fall_back_action.name} as fallback option if no action can be matched.")

        available_actions = get_available_actions(tools)

        # Initialize fact-checker modules
        self.planner = Planner(valid_actions=available_actions,
                               llm=self.llm,
                               logger=self.logger,
                               extra_rules=extra_plan_rules,
                               fall_back=self.fall_back_action)

        self.actor = Actor(tools=tools, llm=self.llm, logger=self.logger)

        self.judge = Judge(llm=self.llm,
                           logger=self.logger,
                           classes=classes,
                           class_definitions=class_definitions,
                           extra_rules=extra_judge_rules)

        self.doc_summarizer = DocSummarizer(self.llm, self.logger)

        self.extra_prepare_rules = extra_prepare_rules
        self.max_iterations = max_iterations

    def _initialize_tools(self, search_engines: list[str]) -> list[Tool]:
        """Loads a default collection of tools."""
        tools = [
            Searcher(search_engines, logger=self.logger),
            CredibilityChecker(logger=self.logger)
        ]

        multimodal = self.mllm is not None
        if multimodal:
            tools.extend([
                ObjectDetector(logger=self.logger),
                Geolocator(top_k=10, logger=self.logger),
                #  TODO: add an image reverse searcher
                FaceRecognizer(logger=self.logger),
                TextExtractor(logger=self.logger)
            ])

        return tools

    def check(self, content: Content | str) -> tuple[Label, list[FCDocument]]:
        """
        Fact-checks the given content ent-to-end by first extracting all check-worthy claims and then
        verifying each claim individually. Returns the aggregated veracity and the list of corresponding
        fact-checking documents, one doc per claim.
        """
        content = Content(content) if isinstance(content, str) else content

        # Input validation
        if content.is_multimodal():
            assert self.mllm is not None, "Multimodal content provided but no multimodal model specified."

        self.logger.log(bold(f"Content to be checked:\n'{light_blue(str(content))}'"), important=True)

        claims = self.claim_extractor.extract_claims(content)

        # Verify each single extracted claim
        self.logger.log(bold("Verifying the claims..."), important=True)
        docs = []
        for claim in claims:  # TODO: parallelize
            doc, evidences = self.verify_claim(claim)
            docs.append(doc)
            self.logger.log(bold(f"The claim '{light_blue(str(claim.text))}' is {doc.verdict.value}."), important=True)
            if doc.justification:
                self.logger.log(f'Justification: {gray(doc.justification)}', important=True)

        overall_veracity = aggregate_predictions([doc.verdict for doc in docs])
        self.logger.log(bold(f"So, the overall veracity is: {overall_veracity.value}"))
        return overall_veracity, docs, evidences

    def verify_claim(self, claim: Claim) -> FCDocument:
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        self.actor.reset()  # remove all past search evidences
        doc = FCDocument(claim)
        initial_evidence = {"claim": claim.text, "evidence":[]}
        questions = self._pose_questions(no_of_questions=10, doc=doc)
        questions_queries_dicts = self.planner.plan_initial_queries(questions, doc)
        self.logger.log(bold(light_blue("Starting Evidence Retrieval for Factuality Questions.")))
        for dict_ in questions_queries_dicts:
            question = dict_["question"]
            self.logger.log(light_blue(f"Answering Question: {question}"))
            actions = self.planner._extract_actions(dict_["queries"], doc.claim.original_context)
            results = [result for action in actions for result in self.actor.get_corresponding_tool_for_action(action).perform(action)]
            self.actor.get_corresponding_tool_for_action(actions[0]).reset()
            generated_answer, url = self.actor.result_summarizer._extract_most_fitting(question, results)
            if not generated_answer or not url:
                continue
            single_evidence = {"question": question, "answer": generated_answer, "url": url}
            initial_evidence["evidence"].append(single_evidence)


        label = Label.NEI
        n_iterations = 0
        while True:
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc)
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            doc.add_actions(actions)
            if not actions:
                break  # the planner wasn't able to determine further useful actions, giving up
            evidences = self.actor.perform(actions, doc)
            doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
            self._consolidate_knowledge(doc, evidences)
            initial_evidence_string = '\n'.join(f'{key}: {value}' for key, value in initial_evidence["evidence"][0].items())
            doc.add_reasoning(initial_evidence_string)
            label = self.judge.judge(doc)
            if label != Label.NEI or n_iterations == self.max_iterations:
                break
            else:
                self.logger.log("Not enough information yet. Continuing fact-check...")
        doc.add_reasoning(self.judge.get_latest_reasoning())
        doc.verdict = label
        if label != Label.REFUSED_TO_ANSWER:
            doc.justification = self.doc_summarizer.summarize(doc)
        return doc, initial_evidence

    def _pose_questions(self, no_of_questions: int, doc: FCDocument):
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc.claim, extra_rules=self.extra_prepare_rules, no_of_questions=no_of_questions)
        answer = self.llm.generate(str(prompt))
        #questions = extract_factuality_questions(answer)
        doc.add_reasoning(answer)
        return answer

    def _consolidate_knowledge(self, doc: FCDocument, evidences: Collection[Evidence]):
        """Analyzes the currently available information and states new questions, adds them
        to the FCDoc."""
        prompt = ReiteratePrompt(doc, evidences)
        answer = self.llm.generate(str(prompt))
        doc.add_reasoning(answer)


def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    veracities = np.array(veracities)
    if np.any(veracities == Label.REFUSED_TO_ANSWER):
        return Label.REFUSED_TO_ANSWER
    elif np.all(veracities == Label.SUPPORTED):
        return Label.SUPPORTED
    elif np.any(veracities == Label.REFUTED):
        return Label.REFUTED
    elif np.any(veracities == Label.CONFLICTING):
        return Label.CONFLICTING
    else:
        return Label.NEI
