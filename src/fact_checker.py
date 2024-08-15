import time
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
from src.prompts.prompt import (PoseQuestionsPrompt, ReiteratePrompt, AnswerCollectively, InterpretPrompt,
                                AnswerQuestion, AnswerQuestionNoEvidence)
from src.tools import *
from src.utils.console import gray, light_blue, bold, sec2mmss
from src.utils.parsing import find_code_span, extract_last_code_span, extract_last_paragraph


class FactChecker:
    """The core class for end-to-end fact verification."""

    def __init__(self,
                 llm: str | LLM = "OPENAI:gpt-3.5-turbo-0125",
                 mllm: Optional[str | MLLM] = None,
                 tools: list[Tool] = None,
                 search_engines: dict[str, dict] = None,
                 procedure_variant: str = "infact",
                 interpret: bool = False,
                 decompose: bool = False,
                 decontextualize: bool = False,
                 filter_check_worthy: bool = False,
                 max_iterations: int = 5,
                 max_result_len: int = None,
                 logger: Logger = None,
                 classes: Sequence[Label] = None,
                 class_definitions: dict[Label, str] = None,
                 extra_prepare_rules: str = None,
                 extra_plan_rules: str = None,
                 extra_judge_rules: str = None,
                 print_log_level: str = "warning",
                 ):
        assert not tools or not search_engines, \
            "You are allowed to specify either tools or search engines."

        self.logger = logger or Logger(print_log_level=print_log_level)

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
        self.max_result_len = max_result_len
        self.procedure_variant = procedure_variant

    def _initialize_tools(self, search_engines: dict[str, dict]) -> list[Tool]:
        """Loads a default collection of tools."""
        tools = [
            Searcher(search_engines, max_result_len=self.max_result_len, logger=self.logger),
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
        start = time.time()

        content = Content(content) if isinstance(content, str) else content

        # Input validation
        if content.is_multimodal():
            assert self.mllm is not None, "Multimodal content provided but no multimodal model specified."

        self.logger.info(bold(f"Content to be checked:\n'{light_blue(str(content))}'"))

        claims = self.claim_extractor.extract_claims(content)

        # Verify each single extracted claim
        self.logger.log(bold("Verifying the claims..."))
        docs = []
        q_and_a = None
        for claim in claims:  # TODO: parallelize
            doc, q_and_a = self.verify_claim(claim)
            docs.append(doc)

        overall_veracity = aggregate_predictions([doc.verdict for doc in docs])
        self.logger.log(bold(f"So, the overall veracity is: {overall_veracity.value}"))
        fc_duration = time.time() - start
        self.logger.log(f"Fact-check took {sec2mmss(fc_duration)}.")
        return overall_veracity, docs, q_and_a

    # --- Procedures starting here ---

    def interpret(self, doc: FCDocument) -> None:
        """Stage 0: Interprets the claim and adds the generated interpretation to the document."""
        prompt = InterpretPrompt(doc.claim)
        response = self.llm.generate(str(prompt))
        doc.add_reasoning(f"## Interpretation\n{response}")

    def perform_naive(self, doc: FCDocument) -> (Label, list):
        verdict = self.judge.judge_naively(doc)
        return verdict, []

    def perform_simple_q_and_a(self, doc: FCDocument) -> (Label, list):
        """InFact but without interpretation and takes first search result."""
        # Stage 1: Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stages 2 & 3: Search query generation and question answering
        q_and_a = self.approach_question_batch(questions, doc)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, q_and_a

    def perform_no_interpretation(self, doc: FCDocument) -> (Label, list):
        """InFact but without interpretation."""
        # Stage 1: Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stages 2 & 3: Search query generation and question answering
        q_and_a = self.approach_question_batch(questions, doc)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, q_and_a

    def perform_no_evidence(self, doc: FCDocument) -> (Label, list):
        """InFact but without any evidence retrieval."""
        # Stage 0: Interpretation generation
        self.interpret(doc)

        # Stage 1: Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stage 3*: Question answering (modified)
        q_and_a = []
        doc.add_reasoning("## Research Q&A")
        for question in questions:
            prompt = AnswerQuestionNoEvidence(question, doc)
            response = self.llm.generate(str(prompt))
            qa_string = (f"### {question}\n"
                         f"Answer: {response}")
            doc.add_reasoning(qa_string)
            qa_instance = {
                "question": question,
                "answer": response,
                "url": "",
                "scraped_text": "",
            }
            q_and_a.append(qa_instance)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, q_and_a

    def perform_first_result(self, doc: FCDocument) -> (Label, list):
        """InFact but using always the first result."""
        # Stage 0: Interpretation generation
        self.interpret(doc)

        # Stage 1: Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stages 2 & 3: Search query generation and question answering
        q_and_a = self.approach_question_batch(questions, doc)  # TODO: modify to accept additional parameter

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, q_and_a

    def perform_no_qa(self, doc: FCDocument) -> (Label, list):
        """InFact but omitting posing any questions."""
        # Stage 0: Interpretation generation
        self.interpret(doc)

        # Stage 2*: Search query generation (modified)
        queries = self.generate_search_queries(doc)

        # Stage 3*: Evidence retrieval (modified)
        results = self.retrieve_search_results(queries, summarize=True, doc=doc)
        doc.add_reasoning("## Web Search")
        for result in results[:10]:
            summary_str = f"### Search Result\n{result}"
            doc.add_reasoning(summary_str)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, []

    def perform_no_query_generation(self, doc: FCDocument) -> (Label, list):
        """InFact but skipping the generation of search queries."""
        # Stage 0: Interpretation generation
        self.interpret(doc)

        # Stage 1: Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stage 3: Evidence retrieval and question answering
        q_and_a = self.approach_question_batch(questions, doc)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, q_and_a

    def perform_q_and_a(self, doc: FCDocument) -> (Label, list):
        """The procedure as implemented by InFact, using all six stages (stage 5, justification
        generation, follows outside of this method)."""
        # Stage 0: Interpretation generation
        self.interpret(doc)

        # Stage 1: Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stages 2 & 3: Search query generation and question answering
        q_and_a = self.approach_question_batch(questions, doc)

        # Stage 4: Veracity prediction
        label = self.judge.judge(doc)

        return label, q_and_a

    def perform_q_and_a_advanced(self, doc: FCDocument) -> (Label, list):
        """The former "dynamic" or "multi iteration" approach."""
        self.interpret(doc)  # stage 0

        # Run iterative Q&A as long as there is NEI
        q_and_a = []
        n_iterations = 0
        label = Label.REFUSED_TO_ANSWER
        while n_iterations < self.max_iterations:
            n_iterations += 1

            questions = self._pose_questions(no_of_questions=4, doc=doc)
            new_qa_instances = self.approach_question_batch(questions, doc)
            q_and_a.extend(new_qa_instances)

            if (label := self.judge.judge(doc)) != Label.NEI:
                break

        # Fill up QA with more questions
        missing_questions = 10 - len(q_and_a)
        if missing_questions > 0:
            questions = self._pose_questions(no_of_questions=missing_questions, doc=doc)
            new_qa_instances = self.approach_question_batch(questions, doc)
            q_and_a.extend(new_qa_instances)

        return label, q_and_a

    def approach_question_batch(self, questions: list[str], doc: FCDocument) -> list:
        """Tries to answer the given list of questions. Unanswerable questions are dropped."""
        # Answer each question, one after another
        q_and_a = []
        doc.add_reasoning("## Research Q&A")
        for question in questions:
            qa_instance = self.approach_question(question, doc)
            if qa_instance is not None:
                q_and_a.append(qa_instance)
                qa_string = (f"### {question}\n"
                             f"Answer: {qa_instance['answer']}\n"
                             f"Source URL: {qa_instance['url']}")
                doc.add_reasoning(qa_string)
        return q_and_a

    def generate_search_queries(self, doc: FCDocument, question: str = None) -> list[WebSearch]:
        return self.planner.propose_queries_for_question(question=question, doc=doc)

    def retrieve_search_results(
            self,
            search_queries: list[WebSearch],
            doc: FCDocument = None,
            summarize: bool = False
    ) -> list[SearchResult]:
        search_results = []
        for query in search_queries:
            evidence = self.actor.perform([query], doc=doc, summarize=summarize)[0]
            search_results.extend(evidence.results)
        return search_results

    def approach_question(self, question: str, doc: FCDocument = None) -> Optional[dict]:
        """Tries to answer the given question. If unanswerable, returns None."""
        self.logger.log(light_blue(f"Answering question: {question}"))
        self.actor.reset()

        # Stage 2: Generate search queries
        match self.procedure_variant:
            case "simple":
                query = self.planner.propose_queries_for_question_simple(question)
                if query is None:
                    self.logger.log("Got no search query, dropping this question.")
                    return None
                queries = [query]
            case "no_query_generation":
                queries = [WebSearch(f'"{question}"')]
            case _:
                queries = self.generate_search_queries(question=question, doc=doc)

        # Execute searches and gather all results
        search_results = self.retrieve_search_results(queries)

        # Step 3: Answer generation
        if len(search_results) > 0:
            return self.generate_answer(question, search_results, doc)

    def generate_answer(self, question: str, results: list[SearchResult], doc: FCDocument) -> Optional[dict]:
        match self.procedure_variant:
            case "advanced":
                answer, relevant_result = self.answer_question_collectively(question, results, doc)
            case "first_result" | "simple":
                relevant_result = results[0]
                answer = self.answer_question(question, relevant_result)
            case _:
                answer, relevant_result = self.answer_question_individually(question, results)

        if relevant_result:
            self.logger.log(f"Got answer: {answer}")
            qa_instance = {"question": question,
                           "answer": answer,
                           "url": relevant_result.source,
                           "scraped_text": relevant_result.text}
            return qa_instance
        else:
            self.logger.log("Got no answer.")

    def perform_open_verification(self, doc: FCDocument) -> (Label, list):
        n_iterations = 0
        while ((label := self.judge.judge(doc)) == Label.NEI
               and n_iterations < self.max_iterations):
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
        return label, []

    # --- Procedures ending here ---

    def verify_claim(self, claim: Claim) -> (FCDocument, dict):
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        self.actor.reset()  # remove all past search evidences
        doc = FCDocument(claim)

        # Depending on the specified procedure variant, perform the fact-check
        match self.procedure_variant:
            case "naive":
                label, q_and_a = self.perform_naive(doc)
            case "no_interpretation":
                label, q_and_a = self.perform_no_interpretation(doc)
            case "no_evidence":
                label, q_and_a = self.perform_no_evidence(doc)
            case "first_result":
                label, q_and_a = self.perform_first_result(doc)
            case "no_qa":
                label, q_and_a = self.perform_no_qa(doc)
            case "no_query_generation":
                label, q_and_a = self.perform_no_query_generation(doc)
            case "infact":
                label, q_and_a = self.perform_q_and_a(doc)
            case "unstructured":
                label, q_and_a = self.perform_open_verification(doc)
            case "simple_qa":
                label, q_and_a = self.perform_simple_q_and_a(doc)
            case "advanced":
                label, q_and_a = self.perform_q_and_a_advanced(doc)  # default FC method
            case _:
                raise ValueError(f"Unknown procedure specified: {self.procedure_variant}")

        # Finalize the fact-check
        doc.add_reasoning("## Final Judgement\n" + self.judge.get_latest_reasoning())

        if label == Label.REFUSED_TO_ANSWER:
            # This part of the code cannot be reached as the judge catches Refused to Answer labels.
            self.logger.warning("The model refused to answer.")
            # label = Label.REFUTED
        else:
            doc.justification = self.doc_summarizer.summarize(doc)
            self.logger.info(bold(f"The claim '{light_blue(str(claim.text))}' is {label.value}."))
            self.logger.info(f'Justification: {gray(doc.justification)}')
        doc.verdict = label

        return doc, q_and_a

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc, n_questions=no_of_questions)
        response = self.llm.generate(str(prompt))
        # Extract the questions
        questions = find_code_span(response)
        return questions

    def answer_question_collectively(
            self,
            question: str,
            results: list[SearchResult],
            doc: FCDocument
    ) -> (Optional[str], Optional[SearchResult]):
        """Generates an answer to the given question by considering batches of 5 search results at once."""
        for i in range(0, len(results), 5):
            results_batch = results[i:i + 5]
            prompt = AnswerCollectively(question, results_batch, doc)
            response = self.llm.generate(str(prompt), max_attempts=3)

            # Extract result ID and answer to the question from response
            if "NONE" not in response and "None" not in response:
                try:
                    result_id = extract_last_code_span(response)
                    if result_id != "":
                        result_id = int(result_id)
                        answer = extract_last_paragraph(response)
                        return answer, results_batch[result_id]
                except:
                    pass
        return None, None

    def answer_question_individually(
            self,
            question: str,
            results: list[SearchResult]
    ) -> (Optional[str], Optional[SearchResult]):
        """Generates an answer to the given question by iterating over the search results
        and using them individually to answer the question."""
        for result in results:
            answer = self.answer_question(question, result)
            if answer is not None:
                return answer, result
        return None, None

    def answer_question(self, question: str, result: SearchResult) -> Optional[str]:
        """Generates an answer to the given question."""
        prompt = AnswerQuestion(question, result)
        response = self.llm.generate(str(prompt), max_attempts=3)
        # Extract answer from response
        if "NONE" not in response and "None" not in response:
            try:
                answer = extract_last_paragraph(response)
                return answer
            except:
                pass

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
