# this should be used only if really needed
# global config parameters, usually constants
import logging
from logging import StreamHandler
import sys 

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s')
my_handler = StreamHandler(sys.stderr)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.DEBUG)



MONITOR_NODE = None


def monitor(t, msg):

    logging.info(f"(Day {t}) NODE-MONITOR: Node {MONITOR_NODE} {msg}")

