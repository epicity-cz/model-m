import datetime
import os

from lang.mytypes import Dict, List


# @puml ignore
class Log:
    SILENT = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4

    log_level: int
    timers: Dict
    filename: str

    def __init__(self, filename=None, log_level=WARN):
        self.log_level = log_level
        self.filename = filename
        if filename and os.path.exists(filename):
            os.remove(filename)
        self.timers = {}

    def level(self, level):
        self.log_level = level

    def log(self, info, level=WARN, end='\n'):
        if level <= self.log_level:
            if self.filename:
                with open(self.filename, "a") as file:
                    file.write(info)
                    file.write(end)
            else:
                print(f"{self.indent()} {info}", end=end)

    def delimited(self, data: List, delim: str = ',', level=WARN, end='\n'):
        self.log(delim.join(map(str, data)), level, end)

    def indent(self):
        return '    ' * len(self.timers)

    def start(self, timer, ll=WARN):
        self.log(f'{timer} starting', ll)
        self.timers[timer] = datetime.datetime.now()

    def stop(self, timer, ll=WARN):
        elapsed = datetime.datetime.now() - self.timers[timer]
        del self.timers[timer]
        self.log(f'{timer} finished, elapsed {elapsed}', ll)


StdLog = Log()
LogSilent = Log(log_level=Log.SILENT)


# @puml ignore
class Timer:
    def __init__(self, info, logger: Log = StdLog, ll=Log.WARN):
        self.info = info
        self.ll = ll
        self.logger = logger

    def __enter__(self):
        self.logger.start(self.info, self.ll)

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.stop(self.info, self.ll)
