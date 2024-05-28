import json
import csv
import logging
import os.path
from datetime import datetime
from logging import handlers
from common.shared_config import path_to_result
from common.label import Label

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.ERROR)


class EvaluationLogger:
    """Used to permanently save any information related to an evaluation run."""

    def __init__(self, dataset_abbr, model_abbr):
        """Initializes the three loggers used for evaluation."""
        log_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Determine the target dir
        self.target_dir = path_to_result + f'{log_date}_{dataset_abbr}_{model_abbr}/'
        os.makedirs(self.target_dir, exist_ok=True)

        # Define file paths
        self.config_path = self.target_dir + 'config.json'
        self.print_path = self.target_dir + 'print.txt'
        self.predictions_path = self.target_dir + 'predictions.csv'
        self.results_path = self.target_dir + 'results.json'

        logging.basicConfig(level=logging.DEBUG)

        self.print_logger = logging.getLogger('print')
        print_handler = handlers.RotatingFileHandler(self.print_path, maxBytes=10 * 1024 * 1024,
                                                     backupCount=5)  # Larger size for print log
        self.print_logger.addHandler(print_handler)
        self.print_logger.propagate = False  # Disable propagation to avoid duplicate logs

        self.results_csv = csv.writer(open(self.predictions_path, "w"))
        self.results_csv.writerow(("sample_index", "target", "predicted", "correct"))

    def save_config(self, config: dict):
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    def log(self, text: str):
        self.print_logger.info(text)

    def save_next_prediction(self, sample_index: int, target: Label, predicted: Label):
        self.results_csv.writerow((sample_index, target, predicted, target == predicted))

    def save_aggregated_results(self, aggregated_results: dict):
        with open(self.results_path, "w") as f:
            json.dump(aggregated_results, f)
