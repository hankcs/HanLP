# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-24 12:47
import os
import warnings
from typing import Union

from hanlp.metrics.f1 import F1_
from hanlp.metrics.mtl import MetricDict
from hanlp.utils.io_util import get_resource, run_cmd, pushd
from hanlp.utils.log_util import flash

_SMATCH_SCRIPT = 'https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced/archive/master.zip#evaluation.sh'
_FAST_SMATCH_SCRIPT = 'https://github.com/jcyk/AMR-gs/archive/master.zip#tools/fast_smatch/compute_smatch.sh'


class SmatchScores(MetricDict):
    @property
    def score(self):
        return self['Smatch'].score


def smatch_eval(pred, gold, use_fast=False) -> Union[SmatchScores, F1_]:
    script = get_resource(_FAST_SMATCH_SCRIPT if use_fast else _SMATCH_SCRIPT)
    home = os.path.dirname(script)
    pred = os.path.realpath(pred)
    gold = os.path.realpath(gold)
    with pushd(home):
        flash('Running evaluation script [blink][yellow]...[/yellow][/blink]')
        cmd = f'bash {script} {pred} {gold}'
        text = run_cmd(cmd)
        flash('')
    return format_fast_scores(text) if use_fast else format_official_scores(text)


def post_process(pred, amr_version):
    pred = os.path.realpath(pred)
    utils_tar_gz = get_amr_utils(amr_version)
    util_dir = get_resource(utils_tar_gz)
    stog_home = get_resource('https://github.com/jcyk/AMR-gs/archive/master.zip')
    with pushd(stog_home):
        run_cmd(
            f'python3 -u -m stog.data.dataset_readers.amr_parsing.postprocess.postprocess '
            f'--amr_path {pred} --util_dir {util_dir} --v 2')
    return pred + '.post'


def get_amr_utils(amr_version):
    if amr_version == '1.0':
        utils_tar_gz = 'https://www.cs.jhu.edu/~s.zhang/data/AMR/amr_1.0_utils.tar.gz'
    elif amr_version == '2.0':
        utils_tar_gz = 'https://www.cs.jhu.edu/~s.zhang/data/AMR/amr_2.0_utils.tar.gz'
    elif amr_version == '3.0':
        utils_tar_gz = 'https://od.hankcs.com/research/amr2020/amr_3.0_utils.tgz'
    else:
        raise ValueError(f'Unsupported AMR version {amr_version}')
    return utils_tar_gz


def format_official_scores(text: str):
    # Smatch -> P: 0.136, R: 0.107, F: 0.120
    # Unlabeled -> P: 0.229, R: 0.180, F: 0.202
    # No WSD -> P: 0.137, R: 0.108, F: 0.120
    # Non_sense_frames -> P: 0.008, R: 0.008, F: 0.008
    # Wikification -> P: 0.000, R: 0.000, F: 0.000
    # Named Ent. -> P: 0.222, R: 0.092, F: 0.130
    # Negations -> P: 0.000, R: 0.000, F: 0.000
    # IgnoreVars -> P: 0.005, R: 0.003, F: 0.003
    # Concepts -> P: 0.075, R: 0.036, F: 0.049
    # Frames -> P: 0.007, R: 0.007, F: 0.007
    # Reentrancies -> P: 0.113, R: 0.060, F: 0.079
    # SRL -> P: 0.145, R: 0.104, F: 0.121
    scores = SmatchScores()
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        name, vs = line.split(' -> ')
        try:
            p, r, f = [float(x.split(': ')[-1]) for x in vs.split(', ')]
        except ValueError:
            warnings.warn(f'Failed to parse results from smatch: {line}')
            p, r, f = float("nan"), float("nan"), float("nan")
        scores[name] = F1_(p, r, f)
    return scores


def format_fast_scores(text: str):
    # using fast smatch
    # Precision: 0.137
    # Recall: 0.108
    # Document F-score: 0.121
    scores = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        name, score = line.split(': ')
        scores.append(float(score))
    assert len(scores) == 3
    return F1_(*scores)
