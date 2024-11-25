import csv
import json
import logging
import os.path
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import shutil

import yaml
import pandas as pd

from config.globals import result_base_dir
from infact.common.document import FCDocument
from infact.common.label import Label
from infact.utils.console import remove_string_formatters, bold, red, orange, yellow, gray
from infact.utils.utils import flatten_dict
from infact.common.medium import media_registry

# Suppress unwanted logging from other libraries
logging.getLogger('bs4').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('git').setLevel(logging.WARNING)
logging.getLogger('wandb').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.ERROR)
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
logging.getLogger('urllib3.connection').setLevel(logging.ERROR)

LOG_LEVELS = {
    "critical": 50,
    "error": 40,
    "warning": 30,
    "info": 20,
    "log": 15,
    "debug": 10,
}


class Logger:
    """Takes care of saving any information (logs, results etc.) related to an evaluation run.

    Args:
        benchmark_name: The shorthand name of the benchmark being evaluated. Used
            to name the directory.
        model_name: The shorthand name of the model used for evaluation. Also used
            to name the directory.
        print_log_level: Pick any of "critical", "error", "warning", "info", "log", "debug"
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
                 experiment_name: str = None,
                 print_log_level: str = "info",
                 target_dir: str | Path = None):
        # Set up the target directory storing all logs and results
        if target_dir:
            self.default_dir = Path(target_dir)
        else:
            self.default_dir = _determine_target_dir(benchmark_name, procedure_name, model_name, experiment_name)

        self._current_fact_check_id = None

        # Define paths for various files
        self.config_path = self.target_dir / self.config_filename
        self.predictions_path = self.target_dir / self.predictions_filename
        self.instance_stats_path = self.target_dir / self.instance_stats_filename
        self.averitec_out = self.target_dir / self.averitec_out_filename
        self.separator = "_" * 25

        logging.basicConfig(level=logging.DEBUG)

        # Initialize the general logger for standard logs
        self.logger = logging.getLogger('mafc')
        self.logger.propagate = False  # Disable propagation to avoid duplicate logs
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(LOG_LEVELS[print_log_level])
        self.logger.addHandler(stdout_handler)
        self.logger.setLevel(logging.DEBUG)

        # Initialize a separate logger for model communication logs
        self.model_comm_logger = logging.getLogger('model_communication')
        self.model_comm_logger.setLevel(logging.DEBUG)
        self.model_comm_logger.propagate = False  # Prevent propagation to the main logger

        self._update_file_handler()

        # Initialize result files (skip if logger is resumed)
        if not os.path.exists(self.predictions_path):
            self._init_predictions_csv()
        if not os.path.exists(self.averitec_out) and benchmark_name == "averitec":
            self._init_averitec_out()

    def set_current_fc_id(self, index: int):
        """Sets the current fact-check ID and initializes related loggers."""
        self._current_fact_check_id = index
        self._update_file_handler()

    def _update_file_handler(self):
        """If a fact-check ID is set, writes to the fact-check-specific folder, otherwise
        to a global log file."""
        # Stop logging into previous fact-check directory
        self._remove_all_file_handlers()

        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Create and add the new file handler for general logs
        general_log_file_handler = _make_file_handler(self.log_path)
        self.logger.addHandler(general_log_file_handler)

        # Create and add the new file handler for model communications
        model_comm_log_file_handler = _make_file_handler(self.model_comm_path)
        self.model_comm_logger.addHandler(model_comm_log_file_handler)

    def _remove_all_file_handlers(self):
        """Removes all existing file handlers from all logger objects."""
        for l in [self.logger, self.model_comm_logger]:
            for handler in l.handlers:
                if isinstance(handler, RotatingFileHandler):
                    self.logger.removeHandler(handler)
                    handler.close()  # Release the file

    def save_config(self, signature, local_scope, benchmark, print_summary: bool = True):
        """Saves the hyperparameters of the current run to a YAML file. Enables to re-use them
        to resume the run."""
        hyperparams = {}
        for param in signature.parameters:
            hyperparams[param] = local_scope[param]
        with open(self.config_path, "w") as f:
            yaml.dump(hyperparams, f)
        if print_summary:
            self.info(bold("Configuration summary:"))
            self.info("Available actions: " + ", ".join([action.name for action in benchmark.available_actions]))
            self.info(yaml.dump(hyperparams, sort_keys=False, indent=4))

    def _init_predictions_csv(self):
        with open(self.predictions_path, "w") as f:
            csv.writer(f).writerow(("sample_index",
                                    "claim",
                                    "target",
                                    "predicted",
                                    "justification",
                                    "correct",
                                    "gt_justification"))

    @property
    def target_dir(self) -> Path:
        if self._current_fact_check_id is None:
            return self.default_dir
        else:
            return self.default_dir / "fact-checks" / str(self._current_fact_check_id)

    @property
    def fc_doc_path(self) -> Path:
        return self.target_dir / "doc.md"

    @property
    def log_path(self) -> Path:
        return self.target_dir / 'log.txt'

    @property
    def model_comm_path(self) -> Path:
        return self.target_dir / 'model_communication.txt'

    def critical(self, msg: str):
        self.logger.critical(bold(red(msg)))

    def error(self, msg: str):
        self.logger.error(red(msg))

    def warning(self, msg: str):
        self.logger.warning(orange(msg))

    def info(self, msg: str):
        self.logger.info(yellow(msg))

    def log(self, msg: str, level=15):
        self.logger.log(level, msg)

    def debug(self, msg: str):
        self.logger.debug(gray(msg))

    def log_model_comm(self, msg: str):
        """Logs model communication using a separate logger."""
        formatted_msg = f"{msg}\n{self.separator}\n\n\n"
        self.model_comm_logger.debug(formatted_msg)

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

    def save_fc_doc(self, doc: FCDocument):
        doc_str = str(doc)

        # Replace all media references with actual file paths for human-readability
        media = media_registry.get_media_from_text(doc_str)
        for medium in media:
            markdown_ref = f"![{medium.data_type} {medium.id}](media/{medium.path_to_file.name})"
            doc_str = doc_str.replace(medium.reference, markdown_ref)

        # Save the markdown file
        with open(self.fc_doc_path, "w") as f:
            f.write(doc_str)

        # Save all associated media in separate media dir
        media_dir = self.target_dir / "media"
        media_dir.mkdir(exist_ok=True)
        for medium in media:
            medium_copy_path = media_dir / medium.path_to_file.name
            shutil.copy(medium.path_to_file, medium_copy_path)

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


def _determine_target_dir(benchmark_name: str,
                          procedure_name: str = None,
                          model_name: str = None,
                          experiment_name: str = None) -> Path:
    assert benchmark_name is not None

    # Construct target directory path
    target_dir = Path(result_base_dir) / benchmark_name

    if procedure_name:
        target_dir /= procedure_name

    if model_name:
        target_dir /= model_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if experiment_name:
        folder_name = f"{timestamp} {experiment_name}"
    else:
        folder_name = timestamp
    target_dir /= folder_name

    # Increment dir name if it exists
    while target_dir.exists():
        target_dir = target_dir.with_stem(timestamp + "'")

    return target_dir


def _make_file_handler(path: Path) -> logging.FileHandler:
    """Sets up a stream that writes all logs with level DEBUG or higher into a dedicated
    TXT file. It automatically removes any string formatting."""
    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    formatter = RemoveStringFormattingFormatter()
    file_handler.setFormatter(formatter)
    return file_handler


if __name__ == "__main__":
    logger = Logger(benchmark_name="test", print_log_level="debug")
    logger.debug("debug")
    logger.log("log")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    logger.log_model_comm("model_comm")
