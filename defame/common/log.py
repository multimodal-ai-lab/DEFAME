import logging
import sys

# Configure the logger to output to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# Create a logger instance that can be imported by other modules
logger = logging.getLogger("DEFAME")