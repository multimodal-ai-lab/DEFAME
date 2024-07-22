from typing import Sequence, Optional, Collection

from src.common.action import *
from src.common.claim import Claim
from src.common.content import Content
from src.common.document import FCDocument
from src.common.label import Label
from src.common.modeling import LLM, MLLM
from src.common.results import Evidence, SearchResult
from src.modules.actor import Actor
from src.modules.claim_extractor import ClaimExtractor
from src.modules.doc_summarizer import DocSummarizer
from src.modules.judge import Judge
from src.modules.planner import Planner
from src.prompts.prompt import PoseQuestionsPrompt, ReiteratePrompt, AnswerPrompt
from src.tools import *
from src.utils.console import gray, light_blue, bold
from src.utils.parsing import find_code_span


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
        assert not tools or not search_engines, \
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

    def check(self, content: Content | str) -> tuple[Label, list[FCDocument], dict]:
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
        q_and_a = None
        for claim in claims:  # TODO: parallelize
            doc, q_and_a = self.verify_claim(claim)
            docs.append(doc)
            self.logger.log(bold(f"The claim '{light_blue(str(claim.text))}' is {doc.verdict.value}."), important=True)
            if doc.justification:
                self.logger.log(f'Justification: {gray(doc.justification)}', important=True)

        overall_veracity = aggregate_predictions([doc.verdict for doc in docs])
        self.logger.log(bold(f"So, the overall veracity is: {overall_veracity.value}"))
        return overall_veracity, docs, q_and_a

    def perform_q_and_a(self, doc: FCDocument) -> list:
        """Asks 10 questions and tries to answer them (required by AVeriTeC challenge)."""
        q_and_a = []

        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Answer each question, one after another
        for question in questions:
            self.actor.reset()
            self.logger.log(light_blue(f"Answering question: {question}"))
            search_actions = self.planner.propose_queries_for_question(question, doc)
            for search_action in search_actions:
                evidence = self.actor.perform([search_action], doc, summarize=False)[0]
                for result in evidence.results:
                    assert isinstance(result, SearchResult)
                    self.logger.log(gray(f"Reading {result.source}"))
                    answer = self.answer_question(question, result, doc)
                    if answer:
                        self.logger.log(f"Got answer: {answer}")
                        q_and_a.append({"question": question,
                                        "answer": answer,
                                        "url": result.source,
                                        "scraped_text": result.text})
                        break
                else:
                    continue  # with query loop if result loop did NOT break
                break  # continue with next question, executes if result loop DID break

        return q_and_a

    def verify_claim(self, claim: Claim) -> (FCDocument, dict):
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        self.actor.reset()  # remove all past search evidences
        doc = FCDocument(claim)

        # Conduct Q&A and insert results into the fact-checking document as initial reasoning
        q_and_a = self.perform_q_and_a(doc)
        q_and_a_strings = [(f"Question: {triplet['question']}\n"
                            f"Answer: {triplet['answer']}\n"
                            f"Source URL: {triplet['url']}") for triplet in q_and_a]
        q_and_a_string = "## Initial Q&A\n" + "\n\n".join(q_and_a_strings)
        doc.add_reasoning(q_and_a_string)

        # Continue the fact-check if Q&A was insufficient
        n_iterations = 0
        while (label := self.judge.judge(doc)) == Label.NEI and n_iterations < self.max_iterations:
            self.logger.log("Not enough information yet. Continuing fact-check...")
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

        doc.add_reasoning(self.judge.get_latest_reasoning())
        doc.verdict = label
        if label == Label.REFUSED_TO_ANSWER:
            # This part of the code cannot be reached as the judge catches Refused to Answer labels.
            self.logger.log("The model refused to answer. We default to Refuted")
            label = Label.REFUTED
        else:
            doc.justification = self.doc_summarizer.summarize(doc)

        return doc, q_and_a

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc.claim, n_questions=no_of_questions)
        response = self.llm.generate(str(prompt))
        # Extract the questions
        questions = find_code_span(response)
        return questions

    def answer_question(self, question: str, result: SearchResult, doc: FCDocument) -> Optional[str]:
        """Generates an answer to the given question."""
        prompt = AnswerPrompt(question, result, doc)
        response = self.llm.generate(str(prompt), max_attempts=3)
        # Validate response
        if len(response) < 16 or "NONE" in response:
            return None
        else:
            return response

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
    elif np.any(veracities == Label.CHERRY_PICKING):
        return Label.CHERRY_PICKING
    else:
        return Label.NEI
