import csv
import json
import logging
import os.path
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from config.globals import result_base_dir
from defame.common.label import Label
from defame.utils.console import remove_string_formatters, bold, red, orange, yellow, gray
from defame.utils.utils import flatten_dict

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
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

LOG_LEVELS = {
    "critical": 50,
    "error": 40,
    "warning": 30,
    "info": 20,
    "log": 15,
    "debug": 10,
}


class Logger:
    """Takes care of saving any information (logs, results etc.) related to an evaluation run."""
    # TODO: Separate general logging tasks from experiment-specific tasks

    log_filename = "log.txt"
    model_comm_filename = "model_communication.txt"
    averitec_out_filename = "averitec_out.json"
    config_filename = "config.yaml"
    predictions_filename = "predictions.csv"
    instance_stats_filename = "instance_stats.csv"

    def __init__(self):
        self.experiment_dir = None
        self._current_fact_check_id: Optional[str] = None
        self.print_log_level = "debug"
        self.connection: Optional[Connection] = None
        self.separator = "_" * 25
        self.is_averitec_run = None

        logging.basicConfig(level=logging.DEBUG)

        # Initialize the general logger for standard logs
        self.logger = logging.getLogger('mafc')
        self.logger.propagate = False  # Disable propagation to avoid duplicate logs
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(LOG_LEVELS[self.print_log_level])
        self.logger.addHandler(stdout_handler)
        self.logger.setLevel(logging.DEBUG)

        # Initialize a separate logger for model communication logs
        self.model_comm_logger = logging.getLogger('model_communication')
        self.model_comm_logger.setLevel(logging.DEBUG)
        self.model_comm_logger.propagate = False  # Prevent propagation to the main logger

    def set_experiment_dir(self,
                           path: str | Path = None,
                           benchmark_name: str = None,
                           procedure_name: str = None,
                           model_name: str = None,
                           experiment_name: str = None):
        """Specify the experiment directory to print the logs and experiment results into.

        Args:
            path: If specified, re-uses an existing directory, i.e., it appends logs
                and results to existing files.
            benchmark_name: The shorthand name of the benchmark being evaluated. Used
                to name the directory.
            procedure_name: THe specifier for the used fact-checking procedure.
            model_name: The shorthand name of the model used for evaluation. Also used
                to name the directory.
            experiment_name: Optional a label to distinguish this experiment run."""
        assert path is not None or benchmark_name is not None

        if path is not None:
            self.experiment_dir = Path(path)
        else:
            self.experiment_dir = _determine_target_dir(benchmark_name, procedure_name, model_name, experiment_name)

        if benchmark_name == "averitec":
            self.is_averitec_run = True

        self._update_file_handler()

    def set_log_level(self, level: str):
        """Pick any of "critical", "error", "warning", "info", "log", "debug"."""
        self.print_log_level = level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(LOG_LEVELS[level])

    def set_current_fc_id(self, identifier: str):
        """Sets the current fact-check ID and initializes related loggers."""
        self._current_fact_check_id = identifier
        self._update_file_handler()

    def set_connection(self, message_conn: Connection):
        self.connection = message_conn

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
                    l.removeHandler(handler)
                    handler.close()  # Release the file

    @property
    def target_dir(self) -> Path:
        if self._current_fact_check_id is None:
            return self.experiment_dir
        else:
            return self.experiment_dir / "fact-checks" / self._current_fact_check_id

    @property
    def config_path(self) -> Path:
        return self.target_dir / self.config_filename

    @property
    def predictions_path(self) -> Path:
        return self.target_dir / self.predictions_filename

    @property
    def instance_stats_path(self) -> Path:
        return self.target_dir / self.instance_stats_filename

    @property
    def averitec_out(self) -> Path:
        return self.target_dir / self.averitec_out_filename

    @property
    def log_path(self) -> Path:
        return self.target_dir / self.log_filename

    @property
    def model_comm_path(self) -> Path:
        return self.target_dir / self.model_comm_filename

    def send(self, msg: str):
        """Sends the message through the connection."""
        if self.connection is not None and self._current_fact_check_id is not None:
            self.connection.send(dict(
                task_id=self._current_fact_check_id,
                status_message=msg,
            ))

    def critical(self, *args, send: bool = True):
        msg = compose_message(*args)
        self.logger.critical(bold(red(msg)))
        if send:
            self.send(msg)

    def error(self, *args, send: bool = True):
        msg = compose_message(*args)
        self.logger.error(red(msg))
        if send:
            self.send(msg)

    def warning(self, *args, send: bool = False):
        msg = compose_message(*args)
        self.logger.warning(orange(msg))
        if send:
            self.send(msg)

    def info(self, *args, send: bool = False):
        msg = compose_message(*args)
        self.logger.info(yellow(msg))
        if send:
            self.send(msg)

    def log(self, *args, level=15, send: bool = False):
        msg = compose_message(*args)
        self.logger.log(level, msg)
        if send:
            self.send(msg)

    def debug(self, *args, send: bool = False):
        msg = compose_message(*args)
        self.logger.debug(gray(msg))
        if send:
            self.send(msg)

    def log_model_comm(self, msg: str):
        """Logs model communication using a separate logger."""
        formatted_msg = f"{msg}\n{self.separator}\n\n\n"
        self.model_comm_logger.debug(formatted_msg)

    def save_config(self, signature, local_scope, print_summary: bool = True):
        """Saves the hyperparameters of the current run to a YAML file. Enables to re-use them
        to resume the run."""
        assert self.experiment_dir is not None
        hyperparams = {}
        for param in signature.parameters:
            hyperparams[param] = local_scope[param]
        with open(self.config_path, "w") as f:
            yaml.dump(hyperparams, f)
        if print_summary:
            self.log(bold("Configuration summary:"))
            self.log(yaml.dump(hyperparams, sort_keys=False, indent=4))

    def _init_predictions_csv(self):
        assert self.experiment_dir is not None
        with open(self.predictions_path, "w") as f:
            csv.writer(f).writerow(("sample_index",
                                    "claim",
                                    "target",
                                    "predicted",
                                    "justification",
                                    "correct",
                                    "gt_justification"))

    def save_next_prediction(self,
                             sample_index: int,
                             claim: str,
                             target: Optional[Label],
                             predicted: Label,
                             justification: str,
                             gt_justification: Optional[str]):
        assert self.experiment_dir is not None

        if not os.path.exists(self.predictions_path):
            self._init_predictions_csv()

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
        assert self.experiment_dir is not None
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

    def _init_averitec_out(self):
        assert self.experiment_dir is not None
        with open(self.averitec_out, "w") as f:
            json.dump([], f, indent=4)

    def save_next_averitec_out(self, next_out: dict):
        assert self.experiment_dir is not None

        if not os.path.exists(self.averitec_out) and self.is_averitec_run:
            self._init_averitec_out()

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


def _determine_target_dir(benchmark_name: str = "testing",
                          procedure_name: str = None,
                          model_name: str = None,
                          experiment_name: str = None) -> Path:
    # assert benchmark_name is not None

    benchmark_name = benchmark_name if benchmark_name else "testing"

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


def compose_message(*args) -> str:
    msg = " ".join([str(a) for a in args])
    return msg.encode("utf-8", "ignore").decode("utf-8")  # remove invalid chars


logger = Logger()
