import logging
import sys


def create_logger(log_file_path):
    log_file = open(log_file_path, "w")
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setLevel("DEBUG")
    log_console_handler = logging.StreamHandler(sys.stderr)
    log_console_handler.setLevel("WARNING")
    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)
    return logger
