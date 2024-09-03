import csv
import json
import logging
import os.path
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd

from config.globals import result_base_dir
from infact.common.document import FCDocument
from infact.common.label import Label
from infact.utils.console import remove_string_formatters, bold, red, orange, yellow
from infact.utils.utils import flatten_dict

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


class Logger:
    """Takes care of saving any information (logs, results etc.) related to an evaluation run.

    Args:
        benchmark_name: The shorthand name of the benchmark being evaluated. Used
            to name the directory.
        model_name: The shorthand name of the model used for evaluation. Also used
            to name the directory.
        print_log_level: Pick any of "critical", "error", "warning", "info", "debug"
        target_dir: If specified, re-uses an existing directory, i.e., it appends logs
            and results to existing files."""

    averitec_out_filename = 'averitec_out.json'
    config_filename = 'config.yaml'
    predictions_filename = 'predictions.csv'
    instance_stats_filename = 'instance_stats.csv'

    def __init__(self,
                 benchmark_name: str = None,
                 procedure_name: str = None,
                 model_name: str = None,
                 print_log_level: str = "warning",
                 target_dir: str | Path = None):
        # TODO: Enable Logger to have NO target dir

        # Set up the target directory storing all logs and results
        self.target_dir = _determine_target_dir(benchmark_name, procedure_name, model_name) \
            if target_dir is None else Path(target_dir)
        os.makedirs(self.target_dir, exist_ok=True)

        # Define any other file and directory paths
        self.fc_docs_dir = self.target_dir / 'docs'
        os.makedirs(self.fc_docs_dir, exist_ok=True)
        self.logs_dir = self.target_dir / 'logs'
        os.makedirs(self.logs_dir, exist_ok=True)

        self.config_path = self.target_dir / self.config_filename
        self.predictions_path = self.target_dir / self.predictions_filename
        self.instance_stats_path = self.target_dir / self.instance_stats_filename
        self.averitec_out = self.target_dir / self.averitec_out_filename

        logging.basicConfig(level=logging.DEBUG)

        self._current_fact_check_id = None

        # Initialize logging module for both console printing and file logging
        self.logger = logging.getLogger('mafc')
        self.logger.propagate = False  # Disable propagation to avoid duplicate logs
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(print_log_level.upper())
        self.logger.addHandler(stdout_handler)
        self._update_file_handler()

        # Initialize result files (skip if logger is resumed)
        if not os.path.exists(self.predictions_path):
            self._init_predictions_csv()
        if not os.path.exists(self.averitec_out):
            self._init_averitec_out()

    def set_current_fc_id(self, index: int):
        self._current_fact_check_id = index
        self._update_file_handler()

    def _update_file_handler(self):
        """If a fact-check ID is set, writes to logs/<fc_id>.txt, otherwise to log.txt."""
        # Remove all previous file handlers
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                self.logger.removeHandler(handler)
                handler.close()  # release the file

        if self._current_fact_check_id is None:
            log_path = self.target_dir / 'log.txt'
        else:
            log_path = self.logs_dir / f'{self._current_fact_check_id}.txt'

        # Create and add the new file handler
        log_to_file_handler = RotatingFileHandler(log_path,
                                                  maxBytes=10 * 1024 * 1024,
                                                  backupCount=5)
        log_to_file_handler.setLevel(logging.DEBUG)
        formatter = RemoveStringFormattingFormatter()
        log_to_file_handler.setFormatter(formatter)
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

    def critical(self, msg: str):
        self.logger.critical(bold(red(msg)))

    def error(self, msg: str):
        self.logger.error(red(msg))

    def warning(self, msg: str):
        self.logger.warning(orange(msg))

    def info(self, msg: str):
        self.logger.info(yellow(msg))

    def debug(self, msg: str):
        self.logger.debug(msg)

    def log(self, msg: str):
        self.debug(msg)

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

    def save_next_instance_stats(self, stats: dict, claim_id: int):
        all_instance_stats = self._load_stats_df()

        # Convert statistics dict to Pandas dataframe
        instance_stats = flatten_dict(stats)
        instance_stats["ID"] = claim_id
        instance_stats = pd.DataFrame([instance_stats])
        instance_stats.set_index("ID", inplace=True)

        # Append instance stats and save
        all_instance_stats = pd.concat([all_instance_stats, instance_stats])
        all_instance_stats.to_csv(self.instance_stats_path)

    def _load_stats_df(self):
        if os.path.exists(self.instance_stats_path):
            df = pd.read_csv(self.instance_stats_path)
            df.set_index("ID", inplace=True)
            return df
        else:
            return pd.DataFrame()

    def save_fc_doc(self, doc: FCDocument, claim_id: int):
        with open(self.fc_docs_dir / f"{claim_id}.md", "w") as f:
            f.write(str(doc))

    def _init_averitec_out(self):
        with open(self.averitec_out, "w") as f:
            json.dump([], f, indent=4)

    def save_next_averitec_out(self, next_out: dict):
        with open(self.averitec_out, "r") as f:
            current_outs = json.load(f)
        current_outs.append(next_out)
        current_outs.sort(key=lambda x: x["claim_id"])  # Score computation requires sorted output
        with open(self.averitec_out, "w") as f:
            json.dump(current_outs, f, indent=4)


class RemoveStringFormattingFormatter(logging.Formatter):
    """Logging formatter that removes any string formatting symbols from the message."""

    def format(self, record):
        msg = record.getMessage()
        return remove_string_formatters(msg)


def _determine_target_dir(benchmark_name: str, procedure_name: str = None, model_name: str = None) -> Path:
    assert benchmark_name is not None

    # Construct target directory path
    target_dir = Path(result_base_dir) / benchmark_name

    if procedure_name:
        target_dir /= procedure_name

    if model_name:
        target_dir /= model_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    target_dir /= timestamp

    # Increment dir name if it exists
    while target_dir.exists():
        target_dir = target_dir.with_stem(timestamp + "'")

    return target_dir
