import csv
import json
import logging
import os.path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Sequence, Optional

import numpy as np
import yaml

from config.globals import path_to_result
from src.common.document import FCDocument
from src.common.label import Label
from src.utils.console import remove_string_formatters, bold, sec2hhmmss

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
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)


class EvaluationLogger:
    """Used to permanently save any information related to an evaluation run."""

    averitec_out_filename = 'averitec_out.json'

    def __init__(self,
                 dataset_abbr: str = None,
                 model_abbr: str = None,
                 verbose: bool = True,
                 target_dir: str = None):
        """Initializes the files used to track evaluation."""
        log_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Determine the target dir
        if target_dir is None:
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

        # Define file and directory paths
        self.fc_docs_dic = self.target_dir + 'docs/'
        os.makedirs(self.fc_docs_dic, exist_ok=True)
        self.logs_dir = self.target_dir + 'logs/'
        os.makedirs(self.logs_dir, exist_ok=True)

        self.config_path = self.target_dir + 'config.yaml'
        self.predictions_path = self.target_dir + 'predictions.csv'
        self.averitec_out = self.target_dir + self.averitec_out_filename

        logging.basicConfig(level=logging.DEBUG)

        self._current_fact_check_id = None
        self.verbose = verbose

        self.logger = logging.getLogger('print')
        self.logger.propagate = False  # Disable propagation to avoid duplicate logs
        # self.logger.setLevel(verbose * 10)  TODO
        self._update_file_handler()

        # Initialize result files (might exist if this logger is resumed)
        if dataset_abbr and not os.path.exists(self.predictions_path):
            self._init_predictions_csv()

        if not os.path.exists(self.averitec_out):
            self._init_averitec_out()

    def set_current_fc_id(self, index: int):
        self._current_fact_check_id = index

    def _update_file_handler(self):
        """If a fact-check ID is set, writes to logs/<fc_id>.txt, otherwise to log.txt."""
        # Remove all previous file handlers
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                self.logger.removeHandler(handler)
                handler.close()  # release the file

        if self._current_fact_check_id is None:
            log_path = self.target_dir + 'log.txt'
        else:
            log_path = self.logs_dir + f'{self._current_fact_check_id}.txt'

        # Create and add the new file handler
        log_to_file_handler = RotatingFileHandler(log_path,
                                                  maxBytes=10 * 1024 * 1024,
                                                  backupCount=5)
        self.logger.addHandler(log_to_file_handler)

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
                                    "gt_justification"))

    def log(self, text: str, important: bool = False):
        if self.verbose or important:
            print("--> " + text)
        self.logger.info("--> " + remove_string_formatters(text))

    def save_next_prediction(self,
                             sample_index: int,
                             claim: str,
                             target: Optional[Label],
                             predicted: Label,
                             justification: str,
                             gt_justification: Optional[str]):
        target_label_str = target.name if target is not None else None
        is_correct = target == predicted if target is not None else None
        with open(self.predictions_path, "a") as f:
            csv.writer(f).writerow((sample_index,
                                    claim,
                                    target_label_str,
                                    predicted.name,
                                    justification,
                                    is_correct,
                                    gt_justification))

    def save_fc_doc(self, doc: FCDocument, claim_id: int):
        with open(self.fc_docs_dic + f"{claim_id}.md", "w") as f:
            f.write(str(doc))

    def _init_averitec_out(self):
        with open(self.averitec_out, "w") as f:
            json.dump([], f, indent=4)

    def save_next_averitec_out(self, next_out: dict):
        with open(self.averitec_out, "r") as f:
            current_outs = json.load(f)
        current_outs.append(next_out)
        with open(self.averitec_out, "w") as f:
            json.dump(current_outs, f, indent=4)
