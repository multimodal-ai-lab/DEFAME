"""Runs the experiment as configured by the specified configuration file."""

if __name__ == '__main__':
    from defame.eval.utils import run_experiment
    run_experiment(config_file_path="in/defame.yaml")
