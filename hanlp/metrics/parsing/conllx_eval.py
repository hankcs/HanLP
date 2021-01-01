# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-08 22:35
import tempfile

from hanlp.components.parsers.conll import read_conll
from hanlp.utils.io_util import get_resource, get_exitcode_stdout_stderr

CONLLX_EVAL = get_resource(
    'https://github.com/elikip/bist-parser/archive/master.zip' + '#bmstparser/src/utils/eval.pl')


def evaluate(gold_file, pred_file):
    """Evaluate using official CoNLL-X evaluation script (Yuval Krymolowski)

    Args:
      gold_file(str): The gold conllx file
      pred_file(str): The pred conllx file

    Returns:

    
    """
    gold_file = get_resource(gold_file)
    fixed_pred_file = tempfile.NamedTemporaryFile().name
    copy_cols(gold_file, pred_file, fixed_pred_file, keep_comments=False)
    if gold_file.endswith('.conllu'):
        fixed_gold_file = tempfile.NamedTemporaryFile().name
        copy_cols(gold_file, gold_file, fixed_gold_file, keep_comments=False)
        gold_file = fixed_gold_file

    exitcode, out, err = get_exitcode_stdout_stderr(f'perl {CONLLX_EVAL} -q -b -g {gold_file} -s {fixed_pred_file}')
    if exitcode:
        raise RuntimeError(f'eval.pl exited with error code {exitcode} and error message {err} and output {out}.')
    lines = out.split('\n')[-4:]
    las = int(lines[0].split()[3]) / int(lines[0].split()[5])
    uas = int(lines[1].split()[3]) / int(lines[1].split()[5])
    return uas, las


def copy_cols(gold_file, pred_file, copied_pred_file, keep_comments=True):
    """Copy the first 6 columns from gold file to pred file

    Args:
      gold_file: 
      pred_file: 
      copied_pred_file: 
      keep_comments:  (Default value = True)

    Returns:

    
    """
    with open(copied_pred_file, 'w') as to_out, open(pred_file) as pred_file, open(gold_file) as gold_file:
        for idx, (p, g) in enumerate(zip(pred_file, gold_file)):
            while p.startswith('#'):
                p = next(pred_file)
            if not g.strip():
                if p.strip():
                    raise ValueError(
                        f'Prediction file {pred_file.name} does not end a sentence at line {idx + 1}\n{p.strip()}')
                to_out.write('\n')
                continue
            while g.startswith('#') or '-' in g.split('\t')[0]:
                if keep_comments or g.startswith('-'):
                    to_out.write(g)
                g = next(gold_file)
            to_out.write('\t'.join(str(x) for x in g.split('\t')[:6] + p.split('\t')[6:]))
