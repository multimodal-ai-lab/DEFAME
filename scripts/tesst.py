from pathlib import Path
from defame.fact_checker import FactChecker
from defame.common import logger

def main():
    outdir = Path(r"D:\BA\fc_init_debug")
    outdir.mkdir(parents=True, exist_ok=True)
    logger.set_experiment_dir(outdir)
    try:
        fc = FactChecker(
            llm="gpt_4o",
            tools_config={"searcher": {"search_config": {"google": {}}, "limit_per_search": 3}},
            device=None,
            procedure_variant="summary/static",
            interpret=True,
            max_iterations=3,
            max_result_len=64000,
        )
        print("[OK] FactChecker constructed.")
    except Exception as e:
        import traceback
        print("[X] FactChecker failed to construct:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
