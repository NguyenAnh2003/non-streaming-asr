# custom_logger.py
import logging


def setup_logger(name, log_file, level=logging.DEBUG):
    """Function to setup a logger; creates console and file handlers with specified level and format."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    # Set level for handlers
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    # Create formatters and add them to the handlers
    console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger