# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-16 18:44
import os

from hanlp.utils.io_util import get_resource, get_exitcode_stdout_stderr, run_cmd


def official_conll_05_evaluate(pred_path, gold_path):
    script_root = get_resource('http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz')
    lib_path = f'{script_root}/lib'
    if lib_path not in os.environ.get("PERL5LIB", ""):
        os.environ['PERL5LIB'] = f'{lib_path}:{os.environ.get("PERL5LIB", "")}'
    bin_path = f'{script_root}/bin'
    if bin_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f'{bin_path}:{os.environ.get("PATH", "")}'
    eval_info_gold_pred = run_cmd(f'perl {script_root}/bin/srl-eval.pl {gold_path} {pred_path}')
    eval_info_pred_gold = run_cmd(f'perl {script_root}/bin/srl-eval.pl {pred_path} {gold_path}')
    conll_recall = float(eval_info_gold_pred.strip().split("\n")[6].strip().split()[5]) / 100
    conll_precision = float(eval_info_pred_gold.strip().split("\n")[6].strip().split()[5]) / 100
    if conll_recall + conll_precision > 0:
        conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
    else:
        conll_f1 = 0
    return conll_precision, conll_recall, conll_f1


def run_perl(script, src, dst=None):
    os.environ['PERL5LIB'] = f''
    exitcode, out, err = get_exitcode_stdout_stderr(
        f'perl -I{os.path.expanduser("~/.local/lib/perl5")} {script} {src}')
    if exitcode:
        # cpanm -l ~/.local namespace::autoclean
        # cpanm -l ~/.local Moose
        # cpanm -l ~/.local MooseX::SemiAffordanceAccessor module
        raise RuntimeError(err)
    with open(dst, 'w') as ofile:
        ofile.write(out)
    return dst
