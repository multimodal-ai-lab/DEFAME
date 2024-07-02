from eval.evaluate import evaluate
import warnings
warnings.filterwarnings("ignore")

evaluate(
    model="gpt_4o",
    multimodal_model="llava_15",
    model_kwargs=dict(temperature=0.01),
    benchmark_name="verite",
    search_engine="duckduckgo",
    benchmark_kwargs=dict(variant="dev"),
    n_samples=20,
    max_results_per_search=3,
    extract_claims=False,
    verbose=True,
    random_sampling=False,
)
