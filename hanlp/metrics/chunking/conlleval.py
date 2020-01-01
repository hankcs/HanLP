#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported
import io
import sys
import re

from collections import defaultdict, namedtuple
from typing import Tuple, Union

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass


Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.correct_tags = 0  # number of correct chunk tags
        self.found_correct = 0  # number of chunks in corpus
        self.found_guessed = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

    @property
    def states(self):
        return (self.t_correct_chunk, self.t_found_correct, self.t_found_guessed)

    def reset_state(self):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.correct_tags = 0  # number of correct chunk tags
        self.found_correct = 0  # number of chunks in corpus
        self.found_guessed = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)
        for state in self.states:
            state.clear()


class CoNLLEval(object):

    def __init__(self) -> None:
        super().__init__()
        self.count = EvalCounts()

    def reset_state(self):
        self.count.reset_state()

    def update_state(self, true_seqs, pred_seqs):
        count = evaluate(true_seqs, pred_seqs)
        self.count.correct_chunk += count.correct_chunk
        self.count.correct_tags += count.correct_tags
        self.count.found_correct += count.found_correct
        self.count.found_guessed += count.found_guessed
        self.count.token_counter += count.token_counter
        for s, n in zip(self.count.states, count.states):
            for k, v in n.items():
                s[k] = s.get(k, 0) + v

    def result(self, full=True, verbose=True) -> Union[Tuple[Metrics, dict, str], Metrics]:
        if full:
            out = io.StringIO()
            overall, by_type = report(self.count, out)
            text = out.getvalue()
            if verbose:
                print(text)
            out.close()
            return overall, by_type, text
        else:
            overall, _ = metrics(self.count)
            return overall


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)


def evaluate(true_seqs, pred_seqs):
    counts = EvalCounts()
    in_correct = False  # currently processed chunks is correct until now
    last_correct = 'O'  # previous chunk tag in corpus
    last_correct_type = ''  # type of previously identified chunk tag
    last_guessed = 'O'  # previously identified chunk tag
    last_guessed_type = ''  # type of previous chunk tag in corpus

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):

        guessed, guessed_type = split_tag(pred_tag)
        correct, correct_type = split_tag(true_tag)

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                    last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if correct == guessed and guessed_type == correct_type:
            counts.correct_tags += 1
        counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed - correct, total - correct
    p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100. * c.correct_tags / c.token_counter))
        out.write('precision: %6.2f%%; ' % (100. * overall.prec))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100. * m.prec))
        out.write('recall: %6.2f%%; ' % (100. * m.rec))
        out.write('FB1: %6.2f  %d\n' % (100. * m.fscore, c.t_found_guessed[i]))
    return overall, by_type


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    return ((prev_tag == "B" and tag == "B") or
            (prev_tag == "B" and tag == "O") or
            (prev_tag == "I" and tag == "B") or
            (prev_tag == "I" and tag == "O") or

            (prev_tag == "E" and tag == "E") or
            (prev_tag == "E" and tag == "I") or
            (prev_tag == "E" and tag == "O") or
            (prev_tag == "I" and tag == "O") or

            (prev_tag != "O" and prev_tag != "." and prev_type != type_) or
            (prev_tag == "]" or prev_tag == "["))


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunkStart = ((prev_tag == "B" and tag == "B") or
                  (prev_tag == "B" and tag == "B") or
                  (prev_tag == "I" and tag == "B") or
                  (prev_tag == "O" and tag == "B") or
                  (prev_tag == "O" and tag == "I") or

                  (prev_tag == "E" and tag == "E") or
                  (prev_tag == "E" and tag == "I") or
                  (prev_tag == "O" and tag == "E") or
                  (prev_tag == "O" and tag == "I") or

                  (tag != "O" and tag != "." and prev_type != type_) or
                  (tag == "]" or tag == "["))
    # corrected 1998-12-22: these chunks are assumed to have length 1

    # print("startOfChunk?", prevTag, tag, prevType, type)
    # print(chunkStart)
    return chunkStart


def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file, encoding='utf-8') as f:
            counts = evaluate(f, args)
    report(counts)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
