import time

import yaml

""" yaml load configs """


def get_configs(path: str):
    """ return defined configs in yaml file """
    params = yaml.safe_load(open(path, 'r', encoding='utf-8'))
    return params

def get_executing_time(start_time):
    """
    start time as param passing with time.time()
    based on this start time to calculate the executing time of
    a function
    :param start_time
    :return executing time
    """
    end_time = time.time() # calling current time
    result = end_time - start_time # executing time
    return result