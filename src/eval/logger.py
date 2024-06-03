import csv
import json
import yaml
import logging
import os.path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from common.label import Label
from common.shared_config import path_to_result
from common.console import remove_string_formatters, bold

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('duckduckgo_search').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class EvaluationLogger:
    """Used to permanently save any information related to an evaluation run."""

    def __init__(self, dataset_abbr: str = None, model_abbr: str = None, verbose: bool = True):
        """Initializes the three loggers used for evaluation."""
        log_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Determine the target dir
        self.target_dir = path_to_result + log_date
        if dataset_abbr:
            self.target_dir += f'_{dataset_abbr}'
        if model_abbr:
            self.target_dir += f'_{model_abbr}'
        self.target_dir += '/'
        os.makedirs(self.target_dir, exist_ok=True)

        # Define file paths
        self.config_path = self.target_dir + 'config.yaml'
        self.print_path = self.target_dir + 'print.txt'
        self.predictions_path = self.target_dir + 'predictions.csv'
        self.results_path = self.target_dir + 'results.json'

        logging.basicConfig(level=logging.DEBUG)

        self.print_logger = logging.getLogger('print')
        print_handler = RotatingFileHandler(self.print_path,
                                            maxBytes=10 * 1024 * 1024,
                                            backupCount=5)
        self.print_logger.addHandler(print_handler)
        self.print_logger.propagate = False  # Disable propagation to avoid duplicate logs

        self.results_csv = csv.writer(open(self.predictions_path, "w"))
        self.results_csv.writerow(("sample_index", "target", "predicted", "correct"))

        self.verbose = verbose

    def save_config(self, signature, local_scope, print_summary=True):
        hyperparams = {}
        for param in signature.parameters:
            hyperparams[param] = local_scope[param]
        with open(self.config_path, "w") as f:
            yaml.dump(hyperparams, f)
        if print_summary:
            print("Configuration summary:")
            for k, v in hyperparams.items():
                print(f"\t{bold(str(k))}: {v}")

    def log(self, text: str):
        if self.verbose:
            print("--> " +text)
        self.print_logger.info("--> " + remove_string_formatters(text))

    def save_next_prediction(self, sample_index: int, target: Label, predicted: Label):
        self.results_csv.writerow((sample_index, target.name, predicted.name, target == predicted))

    def save_aggregated_results(self, aggregated_results: dict):
        with open(self.results_path, "w") as f:
            json.dump(aggregated_results, f)


