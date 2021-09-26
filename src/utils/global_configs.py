import logging
from logging import StreamHandler
import sys


# this should be used only if really needed
# global config parameters, usually constants


MONITOR_NODE = None
SAVE_NODES = False
SAVE_DURATIONS = False


def monitor(t, msg):
    logging.info(f"(Day {t}) NODE-MONITOR: Node {MONITOR_NODE} {msg}")
