# this should be used only if really needed
# global config parameters, usually constants
import logging
from logging import StreamHandler
import sys

# LOGGING_LEVEL = None

# log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s')
# my_handler = StreamHandler(sys.stderr)
# my_handler.setFormatter(log_formatter)
# my_handler.setLevel(LOGGING_LEVEL if LOGGING_LEVEL is not None else logging.CRITICAL)

# def set_logging(level):
#     LOGGING_LEVEL = level
#     my_handler.setLevel(level)

# def logging_setup():
#     logger = logging.getLogger()
#     if LOGGING_LEVEL is not None:
#         logger.setLevel(LOGGING_LEVEL)
#     logger.addHandler(my_handler)


MONITOR_NODE = None
SAVE_NODES = False
SAVE_DURATIONS = False

def monitor(t, msg):

    logging.info(f"(Day {t}) NODE-MONITOR: Node {MONITOR_NODE} {msg}")
