# -*-coding:utf-8 -*-
import os
import logging


def getLogger(name, log_dir=None):
    """
    Default logger & model specific file logger to write params and text evaluation
    """
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_dir:
        # if log dir not provided, use stream handler only
        handler1 = logging.FileHandler(os.path.join(log_dir, 'train.log'), 'a')
        handler1.setFormatter(formater)
        logger.addHandler(handler1)
    handler2 = logging.StreamHandler()
    handler2.setFormatter(formater)
    logger.addHandler(handler2)
    return logger

logger = getLogger('default')