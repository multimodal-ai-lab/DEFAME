"""Minimal running example for a multimodal fact-check."""

from defame.fact_checker import FactChecker
from defame.common import Image


fact_checker = FactChecker(llm="llama4_scout")
claim = ["The image",
         Image("in/example/sahara.webp"),
         "shows the Sahara in 2023 covered with snow!"]
report, _ = fact_checker.verify_claim(claim)
report.save_to("out/fact-check")
