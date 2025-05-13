import gc
import time
import torch
import traceback

from defame.common import logger
from defame.common.modeling import make_model 
from defame.eval import load_benchmark
from defame.fact_checker import FactChecker
from defame.tools import initialize_tools
from defame.eval.evaluate import process_output, finalize_evaluation

def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def evaluate_sequential(
        llm: str,
        benchmark_name: str,
        tools_config: dict,
        experiment_name: str = None,
        fact_checker_kwargs: dict = None,
        llm_kwargs: dict = None,
        benchmark_kwargs: dict = None,
        allowed_actions: list[str] = None,
        n_samples: int = None,
        sample_ids: list[int] = None,
        random_sampling: bool = False,
        print_log_level: str = "info",
        claims_per_batch: int = 1,  # Number of claims to process before reinitializing
):
    """Sequential evaluation for single-worker setups."""
    print(f"Starting sequential evaluation on {benchmark_name}...")
    
    # Setup benchmark
    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs or {})
    is_test = benchmark.variant == "test"
    
    # Setup logging
    if fact_checker_kwargs is None or "procedure_variant" not in fact_checker_kwargs:
        procedure_variant = FactChecker.default_procedure
    else:
        procedure_variant = fact_checker_kwargs["procedure_variant"]
        
    logger.set_experiment_dir(
        benchmark_name=benchmark.shorthand,
        procedure_name=procedure_variant,
        model_name=llm,
        experiment_name=experiment_name
    )
    logger.set_log_level(print_log_level)
    
    # Determine which claims to process
    if random_sampling:
        benchmark.shuffle()
        
    if n_samples:
        samples = benchmark[:n_samples]
    elif sample_ids:
        samples = [benchmark.get_by_id(i) for i in sample_ids]
    else:
        samples = benchmark
        
    n_samples = len(samples)
    print(f"Processing {n_samples} claims sequentially...")
    
    # Setup timing
    start_time = time.time()
    
    # Initialize model and tools
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model = make_model(llm, device=device, **(llm_kwargs or {}))
    tools = initialize_tools(tools_config or {}, llm_model, device=device)
    
    # Setup fact checker
    fc = FactChecker(
        llm=llm_model,
        tools=tools,
        available_actions=allowed_actions or benchmark.available_actions,
        class_definitions=benchmark.class_definitions,
        extra_prepare_rules=benchmark.extra_prepare_rules,
        extra_plan_rules=benchmark.extra_plan_rules,
        extra_judge_rules=benchmark.extra_judge_rules,
        **(fact_checker_kwargs or {})
    )
    
    # Process claims
    stats = {"Number of workers": 1, "Total run duration": 0}
    claims_in_batch = 0
    
    for i, instance in enumerate(samples):
        content = instance["content"]
        print(f"Processing claim {i+1}/{n_samples}: {content.id_number}")
        
        try:
            # Process the claim
            logger.set_current_fc_id(content.id_number)
            _, docs, metas = fc.check_content(content)
            doc = docs[0]
            meta = metas[0]
            
            # Save results
            process_output(doc, meta, benchmark, is_test)
            
            # Clean up memory
            cleanup_memory()
            
            # Check if we need to reinitialize
            claims_in_batch += 1
            if claims_in_batch >= claims_per_batch:
                print(f"Reinitializing model after processing {claims_per_batch} claims...")
                
                # Delete and reinitialize everything
                del fc, llm_model, tools
                cleanup_memory()
                
                llm_model = make_model(llm, device=device, **(llm_kwargs or {}))
                tools = initialize_tools(tools_config or {}, llm_model, device=device)
                fc = FactChecker(
                    llm=llm_model,
                    tools=tools,
                    available_actions=allowed_actions or benchmark.available_actions,
                    class_definitions=benchmark.class_definitions,
                    extra_prepare_rules=benchmark.extra_prepare_rules,
                    extra_plan_rules=benchmark.extra_plan_rules,
                    extra_judge_rules=benchmark.extra_judge_rules,
                    **(fact_checker_kwargs or {})
                )
                claims_in_batch = 0
                
        except Exception as e:
            error_message = f"Error processing claim {content.id_number}:\n"
            error_message += traceback.format_exc()
            logger.error(error_message)
    
    # Finalize and generate summary statistics
    end_time = time.time()
    stats["Total run duration"] = end_time - start_time
    
    finalize_evaluation(stats, logger.target_dir, benchmark)
    logger.info(f"Sequential evaluation complete. Results saved in {logger.target_dir}.")
