# coding:utf-8
# the code is migrated from https://github.com/SapienzaNLP/spring 
# MIT License
#
# Copyright (c) 2022 xfbai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
from pathlib import Path
from typing import List, Union, Iterable
from hanlp.components.amr.amrbart.preprocess.penman_interface import load as pm_load


def read_raw_amr_data(
        paths: List[Union[str, Path]], use_recategorization=False, dereify=True, remove_wiki=False,
):
    """ code for loading AMR from a set of files
        - use_recategorization: use graph recategorization trick
        - dereify: Dereify edges in g that have reifications in model.
        - remove_wiki: remove wiki links
    """
    assert paths
    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = []
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)
            graphs.extend(pm_load(path, dereify=dereify, remove_wiki=remove_wiki))

    assert graphs

    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata["snt_orig"] = metadata["snt"]
            tokens = eval(metadata["tokens"])
            metadata["snt"] = " ".join(
                [
                    t
                    for t in tokens
                    if not ((t.startswith("-L") or t.startswith("-R")) and t.endswith("-"))
                ]
            )

    return graphs
