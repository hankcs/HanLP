#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import sys
from collections import namedtuple


# ===============================================================
def sdp_eval(gold_files, sys_files, labeled=False):
    """Modified from https://github.com/tdozat/Parser-v3/blob/2ff4061373e8aac8c962537a6220e1d5b196abf6/scripts/semdep_eval.py
    Dozat claimed "I tested it against the official eval script and it reported identical LF1".

    Args:
      gold_files: 
      sys_files: 
      labeled:  (Default value = False)

    Returns:

    
    """

    correct = 0
    predicted = 0
    actual = 0
    n_tokens = 0
    n_sequences = 0
    current_seq_correct = False
    n_correct_sequences = 0
    current_sent = 0
    if isinstance(gold_files, str):
        gold_files = [gold_files]
    if isinstance(sys_files, str):
        sys_files = [sys_files]

    for gold_file, sys_file in zip(gold_files, sys_files):
        with codecs.open(gold_file, encoding='utf-8') as gf, \
                codecs.open(sys_file, encoding='utf-8') as sf:
            gold_line = gf.readline()
            gold_i = 1
            sys_i = 0
            while gold_line:
                while gold_line.startswith('#'):
                    current_sent += 1
                    gold_i += 1
                    n_sequences += 1
                    n_correct_sequences += current_seq_correct
                    current_seq_correct = True
                    gold_line = gf.readline()
                if gold_line.rstrip() != '':
                    sys_line = sf.readline()
                    sys_i += 1
                    while sys_line.startswith('#') or sys_line.rstrip() == '' or sys_line.split('\t')[0] == '0':
                        sys_line = sf.readline()
                        sys_i += 1

                    gold_line = gold_line.rstrip().split('\t')
                    sys_line = sys_line.rstrip().split('\t')
                    # assert sys_line[1] == gold_line[1], 'Files are misaligned at lines {}, {}'.format(gold_i, sys_i)

                    # Compute the gold edges
                    gold_node = gold_line[8]
                    if gold_node != '_':
                        gold_node = gold_node.split('|')
                        if labeled:
                            gold_edges = set(tuple(gold_edge.split(':', 1)) for gold_edge in gold_node)
                        else:
                            gold_edges = set(gold_edge.split(':', 1)[0] for gold_edge in gold_node)
                    else:
                        gold_edges = set()

                    # Compute the sys edges
                    sys_node = sys_line[8]
                    if sys_node != '_':
                        sys_node = sys_node.split('|')
                        if labeled:
                            sys_edges = set(tuple(sys_edge.split(':', 1)) for sys_edge in sys_node)
                        else:
                            sys_edges = set(sys_edge.split(':', 1)[0] for sys_edge in sys_node)
                    else:
                        sys_edges = set()

                    correct_edges = gold_edges & sys_edges
                    if len(correct_edges) != len(gold_edges):
                        current_seq_correct = False
                    correct += len(correct_edges)
                    predicted += len(sys_edges)
                    actual += len(gold_edges)
                    n_tokens += 1
                    # current_fp += len(sys_edges) - len(gold_edges & sys_edges)
                gold_line = gf.readline()
                gold_i += 1
    # print(correct, predicted - correct, actual - correct)
    Accuracy = namedtuple('Accuracy', ['precision', 'recall', 'F1', 'seq_acc'])
    precision = correct / (predicted + 1e-12)
    recall = correct / (actual + 1e-12)
    F1 = 2 * precision * recall / (precision + recall + 1e-12)
    seq_acc = n_correct_sequences / n_sequences
    return Accuracy(precision, recall, F1, seq_acc)


# ===============================================================
def main():
    """ """

    files = sys.argv[1:]
    n_files = len(files)
    assert (n_files % 2) == 0
    gold_files, sys_files = files[:n_files // 2], files[n_files // 2:]
    UAS = sdp_eval(gold_files, sys_files, labeled=False)
    LAS = sdp_eval(gold_files, sys_files, labeled=True)
    # print(UAS.F1, UAS.seq_acc)
    print('UAS={:0.1f}'.format(UAS.F1 * 100))
    print('LAS={:0.1f}'.format(LAS.F1 * 100))


if __name__ == '__main__':
    main()
