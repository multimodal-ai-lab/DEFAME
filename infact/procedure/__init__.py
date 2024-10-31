from .procedure import Procedure
from .variants.qa_based.advanced import AdvancedQA
from .variants.qa_based.first_result import FirstResult
from .variants.qa_based.infact import InFact
from .variants.naive import NaiveQA
from .variants.qa_based.no_evidence import NoEvidence
from .variants.qa_based.no_interpretation import NoInterpretation
from .variants.qa_based.no_query_gen import NoQueryGeneration
from .variants.qa_based.simple import SimpleQA
from .variants.summary_based.default import DynamicSummary
from .variants.summary_based.all_actions import AllActionsSummary
from .variants.summary_based.no_qa import StaticSummary

PROCEDURE_REGISTRY = {
    # QA-based procedures
    "advanced": AdvancedQA,
    "first_result": FirstResult,
    "infact": InFact,
    "naive": NaiveQA,
    "no_evidence": NoEvidence,
    "no_interpretation": NoInterpretation,
    "no_query_generation": NoQueryGeneration,
    "simple_qa": SimpleQA,

    # Summary-based procedures
    "no_qa": StaticSummary,
    "summary": DynamicSummary,
    "all_actions": AllActionsSummary,
}


def get_procedure(name: str, **kwargs) -> Procedure:
    if name in PROCEDURE_REGISTRY:
        return PROCEDURE_REGISTRY[name](**kwargs)
    else:
        raise ValueError(f"'{name}' is not a valid procedure variant. "
                         f"Please use one of the following: {list(PROCEDURE_REGISTRY.keys())}.")
