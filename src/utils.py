# -*- coding: utf8 -*-
#
import logging
import sys


def get_logger():
    logger = logging.getLogger('biaffine')
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'
    file_handler = logging.FileHandler(filename='run.log')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(logging.Formatter(fmt))
    stream_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = get_logger()
