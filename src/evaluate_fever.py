from eval.evaluate import evaluate

evaluate(
    model="OPENAI:gpt-4o-2024-05-13",
    multimodal_model=None,
    search_engine="wiki_dump",
    benchmark_name="fever",
    benchmark_kwargs=dict(variant="dev"),
    extract_claims=False,
    verbose=True,
)
