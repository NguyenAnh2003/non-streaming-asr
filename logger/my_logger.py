import logging
import os
import time

def setup_logger(name: str = "example"):
    """ setup logger with logging package all log information will stored
    in *.log files format date for logfile or filename """

    # setup filename
    filename = f"./logs/{name}.log"

    # format date logger
    logging.Formatter(datefmt='%Y-%m-%d,%H:%M:%S',fmt='%(asctime)s.%(msecs)03d',)
    # logger basic setup
    logging.basicConfig(filename=filename, encoding='utf-8',
                        level=logging.INFO, format='%(levelname)s:%(message)s')
    # define logger for returning
    logger = logging.getLogger(__name__)

    return logger

if __name__ == "__main__":
    logger = setup_logger("example")
    logger.info(f"-- SETUP LOGGER --")
