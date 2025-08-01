"""
This file contains options and objects for logging application output.
"""

import logging
from logging.config import dictConfig


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    # Python want's a flush method for cleanup but since we don't actually maintain a buffer it's empty.
    def flush(self):
        pass


def logging_def(logfile=None, MBlimit=5):
    # Creates a stream of output to console by default.
    # With logfile set, will also send the output to that file.

    logging_config = dict(
        version=1,
        formatters={
            "f": {
                "format": "%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S UTC",
            }
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": logging.INFO,
            }
        },
        root={
            "handlers": ["h"],
            "level": logging.INFO,
        },
    )

    # FIXME: Doesn't seem like the 'mode':'w' is working as expected. Logfile not overwritten.
    if logfile:
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "f",
            "filename": logfile,
            "maxBytes": MBlimit * 1024 * 1024,
            "mode": "a",  # use 'a' for append
            "backupCount": 2,
        }
        (logging_config["root"]["handlers"]).append("file")

    return logging_config
