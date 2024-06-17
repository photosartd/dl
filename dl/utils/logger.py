import logging


def get_logger(level: int = logging.INFO):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=level)
    return logger