"""Minimal running example for a multimodal fact-check."""

from ezmm import Image

from defame.evidence_retrieval import Searcher
from defame.evidence_retrieval.tools import DeepfakeDetector, ImageFactCheckEvidenceTool, Geolocator
from defame.evidence_retrieval.tools.metadata_verifier import MetadataAuthVerifier
from defame.fact_checker import FactChecker

metadata_verifier = MetadataAuthVerifier()
searcher = Searcher()

fact_checker = FactChecker(
    tools_config=dict(searcher=dict(
            search_config=dict(
                google=dict(),
            ),
            limit_per_search=3
        ),metadata_auth_verifier=dict(),geolocator=dict()
    )
)
claim = [Image("in/fa/faok.jpg"),"Original photo taken directly from stadium : Messi vs Ramos"]
report, _ = fact_checker.verify_claim(claim)

#,metadata_auth_verifier=dict()