# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 23:43
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def cdroot():
    """
    cd to project root, so models are saved in the root folder
    """
    os.chdir(root)
