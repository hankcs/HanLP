# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-25 16:04

import os
import tempfile
from typing import List

from hanlp.metrics.parsing.conllx_eval import copy_cols

from hanlp_common.structure import SerializableDict
from hanlp.metrics.parsing import iwpt20_xud_eval
from hanlp.metrics.parsing.iwpt20_xud_eval import load_conllu_file
from hanlp.utils.io_util import get_resource, get_exitcode_stdout_stderr

UD_TOOLS_ROOT = get_resource(
    'https://github.com/UniversalDependencies/tools/archive/1650bd354bd158c75836cff6650ea35cc9928fc8.zip')

ENHANCED_COLLAPSE_EMPTY_NODES = os.path.join(UD_TOOLS_ROOT, 'enhanced_collapse_empty_nodes.pl')
CONLLU_QUICK_FIX = os.path.join(UD_TOOLS_ROOT, 'conllu-quick-fix.pl')


def run_perl(script, src, dst=None):
    if not dst:
        dst = tempfile.NamedTemporaryFile().name
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


def enhanced_collapse_empty_nodes(src, dst=None):
    return run_perl(ENHANCED_COLLAPSE_EMPTY_NODES, src, dst)


def conllu_quick_fix(src, dst=None):
    return run_perl(CONLLU_QUICK_FIX, src, dst)


def load_conll_to_str(path) -> List[str]:
    with open(path) as src:
        text = src.read()
        sents = text.split('\n\n')
        sents = [x for x in sents if x.strip()]
        return sents


def remove_complete_edges(src, dst):
    sents = load_conll_to_str(src)
    with open(dst, 'w') as out:
        for each in sents:
            for line in each.split('\n'):
                if line.startswith('#'):
                    out.write(line)
                else:
                    cells = line.split('\t')
                    cells[7] = cells[7].split(':')[0]
                    out.write('\t'.join(cells))
                out.write('\n')
            out.write('\n')


def remove_collapse_edges(src, dst):
    sents = load_conll_to_str(src)
    with open(dst, 'w') as out:
        for each in sents:
            for line in each.split('\n'):
                if line.startswith('#'):
                    out.write(line)
                else:
                    cells = line.split('\t')
                    deps = cells[8].split('|')
                    deps = [x.split('>')[0] for x in deps]
                    cells[8] = '|'.join(deps)
                    out.write('\t'.join(cells))
                out.write('\n')
            out.write('\n')


def restore_collapse_edges(src, dst):
    sents = load_conll_to_str(src)
    with open(dst, 'w') as out:
        for each in sents:
            empty_nodes = {}  # head to deps
            lines = each.split('\n')
            tokens = [x for x in lines if not x.startswith('#') and x.split()[0].isdigit()]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    out.write(line)
                else:
                    cells = line.split('\t')
                    deps = cells[8].split('|')
                    for i, d in enumerate(deps):
                        if '>' in d:
                            head, rel = d.split(':', 1)
                            ehead = f'{len(tokens)}.{len(empty_nodes) + 1}'
                            par, cur = rel.split('>', 1)
                            cur = cur.split('>')[0]
                            deps[i] = f'{ehead}:{cur}'
                            empty_nodes[ehead] = f'{head}:{par}'
                    cells[8] = '|'.join(deps)
                    out.write('\t'.join(cells))
                out.write('\n')
            num_tokens = int(line.split('\t')[0])
            assert num_tokens == len(tokens)
            for idx, (ehead, deps) in enumerate(empty_nodes.items()):
                out.write(f'{num_tokens}.{idx + 1}\t' + '_\t' * 7 + deps + '\t_\n')
            out.write('\n')


def evaluate(gold_file, pred_file, do_enhanced_collapse_empty_nodes=False, do_copy_cols=True):
    """Evaluate using official CoNLL-X evaluation script (Yuval Krymolowski)

    Args:
      gold_file(str): The gold conllx file
      pred_file(str): The pred conllx file
      do_enhanced_collapse_empty_nodes:  (Default value = False)
      do_copy_cols:  (Default value = True)

    Returns:

    
    """
    if do_enhanced_collapse_empty_nodes:
        gold_file = enhanced_collapse_empty_nodes(gold_file)
        pred_file = enhanced_collapse_empty_nodes(pred_file)
    if do_copy_cols:
        fixed_pred_file = pred_file.replace('.conllu', '.fixed.conllu')
        copy_cols(gold_file, pred_file, fixed_pred_file)
    else:
        fixed_pred_file = pred_file
    args = SerializableDict()
    args.enhancements = '0'
    args.gold_file = gold_file
    args.system_file = fixed_pred_file
    return iwpt20_xud_eval.evaluate_wrapper(args)


def main():
    print(evaluate('data/iwpt2020/iwpt2020-test-gold/cs.conllu',
                   'data/model/iwpt2020/bert/ens/cs.conllu', do_enhanced_collapse_empty_nodes=True))


if __name__ == '__main__':
    main()
