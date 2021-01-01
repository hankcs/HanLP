import math
import os.path
import re
import subprocess
import tempfile

from hanlp.components.parsers.hpsg import trees
from hanlp.datasets.parsing.ptb import _PTB_HOME
from hanlp.metrics.metric import Metric
from hanlp.utils.io_util import get_resource, run_cmd, pushd
from hanlp.utils.log_util import flash
from hanlp.utils.string_util import ispunct


class FScore(Metric):

    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return f"P: {self.precision:.2%} R: {self.recall:.2%} F1: {self.fscore:.2%}"

    @property
    def score(self):
        return self.fscore

    def __call__(self, pred, gold):
        pass

    def reset(self):
        self.recall = 0
        self.precision = 0
        self.fscore = 0


def get_evalb_dir():
    home = os.path.realpath(os.path.join(get_resource(_PTB_HOME), '../EVALB'))
    evalb_path = os.path.join(home, 'evalb')
    if not os.path.isfile(evalb_path):
        flash(f'Compiling evalb to {home}')
        with pushd(home):
            run_cmd(f'make')
        flash('')
        if not os.path.isfile(evalb_path):
            raise RuntimeError(f'Failed to compile evalb at {home}')
    return home


def evalb(gold_trees, predicted_trees, ref_gold_path=None, evalb_dir=None):
    if not evalb_dir:
        evalb_dir = get_evalb_dir()
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
    assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)

    if os.path.exists(evalb_program_path):
        # evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
        evalb_param_path = os.path.join(evalb_dir, "nk.prm")
    else:
        evalb_program_path = evalb_spmrl_program_path
        evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        assert isinstance(gold_tree, trees.TreebankNode)
        assert isinstance(predicted_tree, trees.TreebankNode)
        gold_leaves = list(gold_tree.leaves())
        predicted_leaves = list(predicted_tree.leaves())
        assert len(gold_leaves) == len(predicted_leaves)
        for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves):
            if gold_leaf.word != predicted_leaf.word:
                # Maybe -LRB- => (
                if ispunct(predicted_leaf.word):
                    gold_leaf.word = predicted_leaf.word
                else:
                    print(f'Predicted word {predicted_leaf.word} does not match gold word {gold_leaf.word}')
        # assert all(
        #     gold_leaf.word == predicted_leaf.word
        #     for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    # DELETE
    # predicted_path = 'tmp_predictions.txt'
    # output_path = 'tmp_output.txt'
    # gold_path = 'tmp_gold.txt'

    with open(gold_path, "w") as outfile:
        if ref_gold_path is None:
            for tree in gold_trees:
                outfile.write("{}\n".format(tree.linearize()))
        else:
            with open(ref_gold_path) as goldfile:
                outfile.write(goldfile.read())

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(tree.linearize()))

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    # print(command)
    subprocess.run(command, shell=True)

    fscore = FScore(math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1)) / 100
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1)) / 100
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1)) / 100
                break

    success = (
            not math.isnan(fscore.fscore) or
            fscore.recall == 0.0 or
            fscore.precision == 0.0)

    if success:
        temp_dir.cleanup()
    else:
        # print("Error reading EVALB results.")
        # print("Gold path: {}".format(gold_path))
        # print("Predicted path: {}".format(predicted_path))
        # print("Output path: {}".format(output_path))
        pass

    return fscore


if __name__ == '__main__':
    print(get_evalb_dir())
