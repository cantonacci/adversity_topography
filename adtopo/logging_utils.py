"""Shared standard-library logging for the analysis scripts.

``get_logger(name)`` returns a logger that writes to stdout *and* to a
timestamped file under ``outputs/logs/``, so every run is durably recorded
rather than only printed to the screen (or a transient Slurm ``.out``). Scripts
keep their thin ``log(msg)`` wrapper delegating to this logger, so existing
call sites and any results-file accumulation are unchanged.
"""
import logging
import sys
from datetime import datetime

from adtopo.config import OUT_DIR

_LOG_DIR = OUT_DIR / 'logs'


def get_logger(name, to_file=True):
    """Return a configured logger writing to stdout and a timestamped log file.

    Configuration is applied once per name (handlers are not duplicated on
    repeated calls), so importing/using it from several modules is safe.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

    if to_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        handler = logging.FileHandler(_LOG_DIR / f'{name}_{stamp}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
    return logger
