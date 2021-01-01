# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:42
import os

from hanlp.utils.io_util import get_resource, split_file
from hanlp.utils.log_util import logger

SIGHAN2005 = 'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip'


def make(train):
    root = get_resource(SIGHAN2005)
    train = os.path.join(root, train.split('#')[-1])
    if not os.path.isfile(train):
        full = train.replace('_90.txt', '.utf8')
        logger.info(f'Splitting {full} into training set and valid set with 9:1 proportion')
        valid = train.replace('90.txt', '10.txt')
        split_file(full, train=0.9, dev=0.1, test=0, names={'train': train, 'dev': valid})
        assert os.path.isfile(train), f'Failed to make {train}'
        assert os.path.isfile(valid), f'Failed to make {valid}'
        logger.info(f'Successfully made {train} {valid}')
