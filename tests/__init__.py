# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 23:43
import os

from tests.resources import project_root


def cdroot():
    """
    cd to project root, so models are saved in the root folder
    """
    os.chdir(project_root)
