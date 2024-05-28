import json
import logging
from datetime import datetime
from logging import handlers


# Custom JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        record_dict = {
            'time': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        return json.dumps(record_dict)


# Function to setup logging
def setup_logging(dataset_abbr, model_abbr):
    log_date = datetime.now().strftime("%Y%m%d-%H%M")

    # Define log file paths
    config_log_path = f'log/config_{log_date}_{dataset_abbr}_{model_abbr}.json'
    testing_log_path = f'log/testing_{log_date}_{dataset_abbr}_{model_abbr}.json'
    print_log_path = f'log/print_{log_date}_{dataset_abbr}_{model_abbr}.json'

    logging.basicConfig(level=logging.DEBUG)

    # Create specific loggers
    config_logger = logging.getLogger('config')
    testing_logger = logging.getLogger('testing')
    print_logger = logging.getLogger('print')

    # Create handlers for each logger
    config_handler = handlers.RotatingFileHandler(config_log_path, maxBytes=1024 * 1024, backupCount=5)
    testing_handler = handlers.RotatingFileHandler(testing_log_path, maxBytes=1024 * 1024, backupCount=5)
    print_handler = handlers.RotatingFileHandler(print_log_path, maxBytes=10 * 1024 * 1024,
                                                 backupCount=5)  # Larger size for print log

    # Create JSON formatters
    json_formatter = JsonFormatter()

    # Attach handlers to loggers and set their formatters
    config_handler.setFormatter(json_formatter)
    testing_handler.setFormatter(json_formatter)
    print_handler.setFormatter(json_formatter)

    config_logger.addHandler(config_handler)
    testing_logger.addHandler(testing_handler)
    print_logger.addHandler(print_handler)

    # Disable propagation to avoid duplicate logs
    config_logger.propagate = False
    testing_logger.propagate = False
    print_logger.propagate = False

    return config_logger, testing_logger, print_logger


# Example usage of the loggers
def log_model_config(config_logger, config):
    config_logger.info(f"Model configuration: {json.dumps(config)}")


def log_testing_result(testing_logger, result):
    testing_logger.info(f"Testing result: {json.dumps(result)}")


def print_log(print_logger, message):
    print_logger.info(message)
