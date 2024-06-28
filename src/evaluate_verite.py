from eval.evaluate import evaluate

evaluate(
    model="llama3_70b",
    multimodal_model="llava_16",
    model_kwargs=dict(temperature=0.01),
    benchmark_name="verite",
    search_engine="wiki_dump",
    benchmark_kwargs=dict(variant="dev"),
    n_samples=2,
    max_results_per_search=3,
    extract_claims=False,
    verbose=True,
    random_sampling=False,
)
