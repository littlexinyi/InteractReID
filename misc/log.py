import logging
import time
import sys
import datetime
import os
from easydict import EasyDict
import yaml

def parse_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config

#root_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def init_logger(log_level, file_path):
    """Initialize internal logger of EasyFL.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-6.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)
   # print(root_path)

    # file_path = os.path.join(root_path, f'{config.model.saved_path}/logs/')
    # os.makedirs(file_path, exist_ok=True)

    current_time = datetime.datetime.now()
    file_path = os.path.join(file_path, current_time.strftime("%Y_%m_%d_%H_%M_%S") + ".log")
    # file_path = os.path.join(file_path, "icfg.log")

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler) 

def get_logger(log_path):
    
    init_logger(logging.INFO, file_path=log_path)
    logger = logging.getLogger(__name__)

    return logger