import csv
import logging
import os.path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Sequence

import numpy as np
import yaml

from common.document import FCDocument
from common.label import Label
from config.globals import path_to_result
from utils.console import remove_string_formatters, bold, sec2hhmmss

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('duckduckgo_search').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('timm.models._builder').setLevel(logging.ERROR)
logging.getLogger('timm.models._hub').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)


class EvaluationLogger:
    """Used to permanently save any information related to an evaluation run."""

    def __init__(self, dataset_abbr: str = None, model_abbr: str = None, verbose: bool = True):
        """Initializes the files used to track evaluation."""
        log_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Determine the target dir
        target_dir = path_to_result
        if dataset_abbr:
            target_dir += f'{dataset_abbr}/'
        target_dir += log_date
        if model_abbr:
            target_dir += f'_{model_abbr}'

        # Increment target dir name if it exists
        target_dir_tmp = target_dir
        i = 1
        while os.path.exists(target_dir_tmp):
            target_dir_tmp = target_dir + f'_{i}'
            i += 1
        target_dir = target_dir_tmp

        self.target_dir = target_dir + '/'
        os.makedirs(self.target_dir, exist_ok=True)

        # Define file paths
        self.config_path = self.target_dir + 'config.yaml'
        self.print_path = self.target_dir + 'print.txt'
        self.predictions_path = self.target_dir + 'predictions.csv'
        self.results_path = self.target_dir + 'results.yaml'
        self.fc_docs_path = self.target_dir + 'docs/'
        os.makedirs(self.fc_docs_path, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG)

        self.print_logger = logging.getLogger('print')
        print_handler = RotatingFileHandler(self.print_path,
                                            maxBytes=10 * 1024 * 1024,
                                            backupCount=5)
        self.print_logger.addHandler(print_handler)
        self.print_logger.propagate = False  # Disable propagation to avoid duplicate logs

        if dataset_abbr:
            self._init_predictions_csv()

        self.verbose = verbose

    def save_config(self, signature, local_scope, print_summary: bool = True):
        hyperparams = {}
        for param in signature.parameters:
            hyperparams[param] = local_scope[param]
        with open(self.config_path, "w") as f:
            yaml.dump(hyperparams, f)
        if print_summary:
            print(bold("Configuration summary:"))
            print(yaml.dump(hyperparams, sort_keys=False, indent=4))

    def _init_predictions_csv(self):
        with open(self.predictions_path, "w") as f:
            csv.writer(f).writerow(("sample_index",
                                    "claim",
                                    "target",
                                    "predicted",
                                    "justification",
                                    "correct",
                                    "justification"))

    def log(self, text: str, important: bool = False):
        if self.verbose or important:
            print("--> " + text)
        self.print_logger.info("--> " + remove_string_formatters(text))

    def save_next_prediction(self,
                             sample_index: int,
                             claim: str,
                             target: Label,
                             predicted: Label,
                             justification: str,
                             gt_justification: str):
        with open(self.predictions_path, "a") as f:
            csv.writer(f).writerow((sample_index,
                                    claim,
                                    target.name,
                                    predicted.name,
                                    justification,
                                    target == predicted,
                                    gt_justification))

    def save_fc_doc(self, doc: FCDocument, claim_id: int):
        with open(self.fc_docs_path + f"{claim_id}.md", "w") as f:
            f.write(str(doc))

    def save_results(self,
                     predictions: Sequence[Label],
                     ground_truth: Sequence[Label],
                     duration: float,
                     total_llm_calls: int,
                     search_summary: dict,
                     print_summary: bool = True) -> bool:
        n_samples = len(predictions)
        n_refused = np.count_nonzero(np.array(predictions) == Label.REFUSED_TO_ANSWER)
        correct_predictions = np.asarray(np.array(predictions) == np.array(ground_truth))
        n_correct_predictions = np.sum(correct_predictions)
        n_wrong_predictions = n_samples - n_correct_predictions - n_refused
        accuracy = n_correct_predictions / (n_samples - n_refused)
        search_summary = ", ".join(f"{searcher}: {n_searches}" for searcher, n_searches in search_summary.items())
        result_summary = {
            "Total samples": n_samples,
            "Correct predictions": int(n_correct_predictions),
            "Wrong predictions": int(n_wrong_predictions),
            "Refused predictions": int(n_refused),
            "Accuracy": f"{accuracy * 100:.1f} %",
            "Run duration": sec2hhmmss(duration),
            "Total LLM calls": total_llm_calls,
            "Total searches": search_summary,
        }
        with open(self.results_path, "w") as f:
            yaml.dump(result_summary, f, sort_keys=False)
        if print_summary:
            print("Results:")
            bold_print_dict(result_summary)
        return accuracy


def bold_print_dict(dictionary: dict):
    for key, value in dictionary.items():
        print(f"\t{bold(str(key))}: {value}")
