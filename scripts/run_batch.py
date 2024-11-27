"""Runs all experiments sequentially as specified by the
config YAMLs inside the specified config directory."""

if __name__ == '__main__':
    from defame.eval.utils import run_multiple_experiments
    run_multiple_experiments(config_dir="config/verite")
