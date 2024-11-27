"""Runs all experiments sequentially as specified through the
config YAMLs inside scripts/experiments/."""

CONFIG_DIR = "config/averitec/"

if __name__ == '__main__':  # evaluation uses multiprocessing
    import os
    from multiprocessing import set_start_method
    from pathlib import Path

    from defame.eval.evaluate import evaluate
    from defame.utils.utils import load_config

    set_start_method("spawn")
    config_dir = Path(CONFIG_DIR)

    while configs := os.listdir(config_dir):
        config_path = config_dir / configs[0]
        experiment_params = load_config(config_path)
        os.remove(config_path)
        evaluate(experiment_name=config_path.stem,
                 **experiment_params)
    print("Finished! No more experiments to run.")
