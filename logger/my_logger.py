import logging
import os
import time

def setup_logger(path: str = "./logs/example.log"):
    """ setup logger with logging package all log information will stored
    in *.log files format date for logfile or filename """

    # format date logger
    logging.Formatter(datefmt='%Y-%m-%d,%H:%M:%S',fmt='%(asctime)s.%(msecs)03d',)
    # logger basic setup
    logging.basicConfig(filename=path, encoding='utf-8',
                        level=logging.INFO, format='%(name)s: %(levelname)s: %(message)s')
    return logging

if __name__ == "__main__":
    logger = setup_logger("example")
    logger.getLogger(__name__)
    logger.info(f"-- SETUP LOGGER --")
