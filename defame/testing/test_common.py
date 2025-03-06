import pytest
from defame.common import logger

def test_logger():
    logger.debug("debug")
    logger.log("log")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    logger.log_model_comm("model_comm")

    logger.log("Next, only log messages with level WARNING or higher:")
    logger.set_log_level("warning")

    logger.debug("debug")
    logger.log("log")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    logger.log_model_comm("model_comm")

    logger.set_experiment_dir("test_experiment/")
    logger.warning("This warning should have been logged into a file under 'test_experiment/'.")
