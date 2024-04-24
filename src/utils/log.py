import logging
from utils import const
from pathlib import Path


def logger_init():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    DIR = Path.cwd()

    file_handler = logging.FileHandler(DIR / const.LOG_PATH)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
