# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 15:37
from abc import abstractmethod
from collections import Counter
from typing import Generator, Tuple, Union, Iterable, Any, List

import tensorflow as tf
import numpy as np
from hanlp.common.structure import SerializableDict
from hanlp.common.transform import Transform
from hanlp.components.parsers.alg import kmeans, randperm, arange, tolist
from hanlp.common.constant import ROOT
from hanlp.common.vocab import Vocab
from hanlp.utils.io_util import get_resource
from hanlp.utils.log_util import logger
from hanlp.utils.string_util import ispunct
from hanlp.utils.util import merge_locals_kwargs


class CoNLLWord(SerializableDict):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, deprel=None, phead=None,
                 pdeprel=None):
        """CoNLL format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf

        Parameters
        ----------
        id : int
            Token counter, starting at 1 for each new sentence.
        form : str
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        cpos : str
            Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
        pos : str
            Fine-grained part-of-speech tag, where the tagset depends on the treebank.
        feats : str
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or an underscore if not available.
        head : Union[int, List[int]]
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        deprel : Union[str, List[str]]
            Dependency relation to the HEAD.
        phead : int
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        pdeprel : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = head
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        if isinstance(self.head, list):
            return '\n'.join('\t'.join(['_' if v is None else v for v in values]) for values in [
                [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                 None if head is None else str(head), deprel, self.phead, self.pdeprel] for head, deprel in
                zip(self.head, self.deprel)
            ])
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  None if self.head is None else str(self.head), self.deprel, self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        return list(f for f in
                    [self.form, self.lemma, self.cpos, self.pos, self.feats, self.head, self.deprel, self.phead,
                     self.pdeprel] if f)


class CoNLLSentence(list):
    def __init__(self, words=None):
        """A list of ConllWord

        Parameters
        ----------
        words : Sequence[ConllWord]
            words of a sentence
        """
        super().__init__()
        if words:
            self.extend(words)

    def __str__(self):
        return '\n'.join([word.__str__() for word in self])

    @staticmethod
    def from_str(conll: str):
        """
        Build a CoNLLSentence from CoNLL-X format str

        Parameters
        ----------
        conll : str
             CoNLL-X format string

        Returns
        -------
        CoNLLSentence

        """
        words: List[CoNLLWord] = []
        prev_id = None
        for line in conll.strip().split('\n'):
            if line.startswith('#'):
                continue
            cells = line.split()
            cells[0] = int(cells[0])
            cells[6] = int(cells[6])
            if cells[0] != prev_id:
                words.append(CoNLLWord(*cells))
            else:
                if isinstance(words[-1].head, list):
                    words[-1].head.append(cells[6])
                    words[-1].deprel.append(cells[7])
                else:
                    words[-1].head = [words[-1].head] + [cells[6]]
                    words[-1].deprel = [words[-1].deprel] + [cells[7]]
            prev_id = cells[0]
        return CoNLLSentence(words)


def read_conll(filepath):
    sent = []
    filepath = get_resource(filepath)
    with open(filepath, encoding='utf-8') as src:
        for line in src:
            if line.startswith('#'):
                continue
            cells = line.strip().split()
            if cells:
                cells[0] = int(cells[0])
                cells[6] = int(cells[6])
                for i, x in enumerate(cells):
                    if x == '_':
                        cells[i] = None
                sent.append(cells)
            else:
                yield sent
                sent = []
    if sent:
        yield sent


class CoNLLTransform(Transform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=True, n_buckets=32,
                 n_tokens_per_batch=5000, min_freq=2,
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.form_vocab: Vocab = None
        self.cpos_vocab: Vocab = None
        self.rel_vocab: Vocab = None
        self.puncts: tf.Tensor = None

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        form, cpos = x
        return self.form_vocab.token_to_idx_table.lookup(form), self.cpos_vocab.token_to_idx_table.lookup(cpos)

    def y_to_idx(self, y):
        head, rel = y
        return head, self.rel_vocab.token_to_idx_table.lookup(rel)

    def X_to_inputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]]) -> Iterable:
        if len(X) == 2:
            form_batch, cposes_batch = X
            mask = tf.not_equal(form_batch, 0)
        elif len(X) == 3:
            form_batch, cposes_batch, mask = X
        else:
            raise ValueError(f'Expect X to be 2 or 3 elements but got {repr(X)}')
        sents = []

        for form_sent, cposes_sent, length in zip(form_batch, cposes_batch,
                                                  tf.math.count_nonzero(mask, axis=-1)):
            forms = tolist(form_sent)[1:length + 1]
            cposes = tolist(cposes_sent)[1:length + 1]
            sents.append([(self.form_vocab.idx_to_token[f],
                           self.cpos_vocab.idx_to_token[c]) for f, c in zip(forms, cposes)])

        return sents

    def lock_vocabs(self):
        super().lock_vocabs()
        self.puncts = tf.constant([i for s, i in self.form_vocab.token_to_idx.items()
                                   if ispunct(s)], dtype=tf.int64)

    def file_to_inputs(self, filepath: str, gold=True):
        assert gold, 'only support gold file for now'
        for sent in read_conll(filepath):
            for i, cell in enumerate(sent):
                form = cell[1]
                cpos = cell[3]
                head = cell[6]
                deprel = cell[7]
                sent[i] = [form, cpos, head, deprel]
            yield sent

    @property
    def bos(self):
        if self.form_vocab.idx_to_token is None:
            return ROOT
        return self.form_vocab.idx_to_token[2]

    def file_to_dataset(self, filepath: str, gold=True, map_x=None, map_y=None, batch_size=5000, shuffle=None,
                        repeat=None, drop_remainder=False, prefetch=1, cache=True, **kwargs) -> tf.data.Dataset:
        return super().file_to_dataset(filepath, gold, map_x, map_y, batch_size, shuffle, repeat, drop_remainder,
                                       prefetch, cache, **kwargs)

    def input_is_single_sample(self, input: Any) -> bool:
        return isinstance(input[0][0], str) if len(input[0]) else False

    def samples_to_dataset(self, samples: Generator, map_x=None, map_y=None, batch_size=5000, shuffle=None, repeat=None,
                           drop_remainder=False, prefetch=1, cache=True) -> tf.data.Dataset:
        if shuffle:
            def generator():
                # custom bucketing, load corpus into memory
                corpus = list(x for x in (samples() if callable(samples) else samples))
                lengths = [self.len_of_sent(i) for i in corpus]
                if len(corpus) < 32:
                    n_buckets = 1
                else:
                    n_buckets = min(self.config.n_buckets, len(corpus))
                buckets = dict(zip(*kmeans(lengths, n_buckets)))
                sizes, buckets = zip(*[
                    (size, bucket) for size, bucket in buckets.items()
                ])
                # the number of chunks in each bucket, which is clipped by
                # range [1, len(bucket)]
                chunks = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1)) for size, bucket in
                          zip(sizes, buckets)]
                range_fn = randperm if shuffle else arange
                max_samples_per_batch = self.config.get('max_samples_per_batch', None)
                for i in tolist(range_fn(len(buckets))):
                    split_sizes = [(len(buckets[i]) - j - 1) // chunks[i] + 1
                                   for j in range(chunks[i])]  # how many sentences in each batch
                    for batch_indices in tf.split(range_fn(len(buckets[i])), split_sizes):
                        indices = [buckets[i][j] for j in tolist(batch_indices)]
                        if max_samples_per_batch:
                            for j in range(0, len(indices), max_samples_per_batch):
                                yield from self.batched_inputs_to_batches(corpus, indices[j:j + max_samples_per_batch],
                                                                          shuffle)
                        else:
                            yield from self.batched_inputs_to_batches(corpus, indices, shuffle)

        else:
            def generator():
                # custom bucketing, load corpus into memory
                corpus = list(x for x in (samples() if callable(samples) else samples))
                n_tokens = 0
                batch = []
                for idx, sent in enumerate(corpus):
                    sent_len = self.len_of_sent(sent)
                    if n_tokens + sent_len > batch_size and batch:
                        yield from self.batched_inputs_to_batches(corpus, batch, shuffle)
                        n_tokens = 0
                        batch = []
                    n_tokens += sent_len
                    batch.append(idx)
                if batch:
                    yield from self.batched_inputs_to_batches(corpus, batch, shuffle)

        # next(generator())
        return Transform.samples_to_dataset(self, generator, False, False, 0, False, repeat, drop_remainder, prefetch,
                                            cache)

    def len_of_sent(self, sent):
        return 1 + len(sent)  # take ROOT into account

    @abstractmethod
    def batched_inputs_to_batches(self, corpus, indices, shuffle):
        """
        Convert batched inputs to batches of samples

        Parameters
        ----------
        corpus : list
            A list of inputs
        indices : list
            A list of indices, each list belongs to a batch

        Returns
        -------
        None

        Yields
        -------
        tuple
            tuple of tf.Tensor
        """
        pass


class CoNLL_DEP_Transform(CoNLLTransform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=True, n_buckets=32,
                 n_tokens_per_batch=5000, min_freq=2, **kwargs) -> None:
        super().__init__(config, map_x, map_y, lower, n_buckets, n_tokens_per_batch, min_freq, **kwargs)

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        types = (tf.int64, tf.int64), (tf.int64, tf.int64)
        shapes = ([None, None], [None, None]), ([None, None], [None, None])
        values = (self.form_vocab.safe_pad_token_idx, self.cpos_vocab.safe_pad_token_idx), (
            0, self.rel_vocab.safe_pad_token_idx)
        return types, shapes, values

    def batched_inputs_to_batches(self, corpus, indices, shuffle):
        """
        Convert batched inputs to batches of samples

        Parameters
        ----------
        corpus : list
            A list of inputs
        indices : list
            A list of indices, each list belongs to a batch

        Returns
        -------
        None

        Yields
        -------
        tuple
            tuple of tf.Tensor
        """
        raw_batch = [[], [], [], []]
        for idx in indices:
            for b in raw_batch:
                b.append([])
            for cells in corpus[idx]:
                for b, c, v in zip(raw_batch, cells,
                                   [self.form_vocab, self.cpos_vocab, None, self.rel_vocab]):
                    b[-1].append(v.get_idx_without_add(c) if v else c)
        batch = []
        for b, v in zip(raw_batch, [self.form_vocab, self.cpos_vocab, None, self.rel_vocab]):
            b = tf.keras.preprocessing.sequence.pad_sequences(b, padding='post',
                                                              value=v.safe_pad_token_idx if v else 0,
                                                              dtype='int64')
            batch.append(b)
        assert len(batch) == 4
        yield (batch[0], batch[1]), (batch[2], batch[3])

    def inputs_to_samples(self, inputs, gold=False):
        for sent in inputs:
            sample = []
            if self.config['lower']:
                for i, cell in enumerate(sent):
                    cell = list(sent[i])
                    cell[0] = cell[0].lower()
                    if not gold:
                        cell += [0, self.rel_vocab.safe_pad_token]
                    sample.append(cell)
            # insert root word with arbitrary fields, anyway it will be masked
            # form, cpos, head, deprel = sample[0]
            sample.insert(0, [self.bos, self.bos, 0, self.bos])
            yield sample

    def XY_to_inputs_outputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]], Y: Union[tf.Tensor, Tuple[tf.Tensor]],
                             gold=False, inputs=None, conll=True) -> Iterable:
        (words, feats, mask), (arc_preds, rel_preds) = X, Y
        if inputs is None:
            inputs = self.X_to_inputs(X)
        ys = self.Y_to_outputs((arc_preds, rel_preds, mask), inputs=inputs)
        sents = []
        for x, y in zip(inputs, ys):
            sent = CoNLLSentence()
            for idx, ((form, cpos), (head, deprel)) in enumerate(zip(x, y)):
                if conll:
                    sent.append(CoNLLWord(id=idx + 1, form=form, cpos=cpos, head=head, deprel=deprel))
                else:
                    sent.append([head, deprel])
            sents.append(sent)
        return sents

    def fit(self, trn_path: str, **kwargs) -> int:
        self.form_vocab = Vocab()
        self.form_vocab.add(ROOT)  # make root the 2ed elements while 0th is pad, 1st is unk
        self.cpos_vocab = Vocab(pad_token=None, unk_token=None)
        self.rel_vocab = Vocab(pad_token=None, unk_token=None)
        num_samples = 0
        counter = Counter()
        for sent in self.file_to_samples(trn_path, gold=True):
            num_samples += 1
            for idx, (form, cpos, head, deprel) in enumerate(sent):
                if idx == 0:
                    root = form
                else:
                    counter[form] += 1
                self.cpos_vocab.add(cpos)
                self.rel_vocab.add(deprel)

        for token in [token for token, freq in counter.items() if freq >= self.config.min_freq]:
            self.form_vocab.add(token)
        return num_samples

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None) -> Iterable:
        arc_preds, rel_preds, mask = Y
        sents = []

        for arc_sent, rel_sent, length in zip(arc_preds, rel_preds,
                                              tf.math.count_nonzero(mask, axis=-1)):
            arcs = tolist(arc_sent)[1:length + 1]
            rels = tolist(rel_sent)[1:length + 1]
            sents.append([(a, self.rel_vocab.idx_to_token[r]) for a, r in zip(arcs, rels)])

        return sents


class CoNLL_SDP_Transform(CoNLLTransform):
    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=True, n_buckets=32,
                 n_tokens_per_batch=5000, min_freq=2, **kwargs) -> None:
        super().__init__(config, map_x, map_y, lower, n_buckets, n_tokens_per_batch, min_freq, **kwargs)
        self.orphan_relation = ROOT

    def lock_vocabs(self):
        super().lock_vocabs()
        # heuristic to find the orphan relation
        for rel in self.rel_vocab.idx_to_token:
            if 'root' in rel.lower():
                self.orphan_relation = rel
                break

    def file_to_inputs(self, filepath: str, gold=True):
        assert gold, 'only support gold file for now'
        for i, sent in enumerate(read_conll(filepath)):
            prev_cells = None
            parsed_sent = []
            heads = []
            rels = []
            for j, cell in enumerate(sent):
                ID = cell[0]
                form = cell[1]
                cpos = cell[3]
                head = cell[6]
                deprel = cell[7]
                if prev_cells and ID != prev_cells[0]:  # found end of token
                    parsed_sent.append([prev_cells[1], prev_cells[2], heads, rels])
                    heads = []
                    rels = []
                heads.append(head)
                rels.append(deprel)
                prev_cells = [ID, form, cpos, head, deprel]
            parsed_sent.append([prev_cells[1], prev_cells[2], heads, rels])
            yield parsed_sent

    def fit(self, trn_path: str, **kwargs) -> int:
        self.form_vocab = Vocab()
        self.form_vocab.add(ROOT)  # make root the 2ed elements while 0th is pad, 1st is unk
        self.cpos_vocab = Vocab(pad_token=None, unk_token=None)
        self.rel_vocab = Vocab(pad_token=None, unk_token=None)
        num_samples = 0
        counter = Counter()
        for sent in self.file_to_samples(trn_path, gold=True):
            num_samples += 1
            for idx, (form, cpos, head, deprel) in enumerate(sent):
                if idx == 0:
                    root = form
                else:
                    counter[form] += 1
                self.cpos_vocab.add(cpos)
                self.rel_vocab.update(deprel)

        for token in [token for token, freq in counter.items() if freq >= self.config.min_freq]:
            self.form_vocab.add(token)
        return num_samples

    def inputs_to_samples(self, inputs, gold=False):
        for sent in inputs:
            sample = []
            if self.config['lower']:
                for i, cell in enumerate(sent):
                    cell = list(sent[i])
                    cell[0] = cell[0].lower()
                    if not gold:
                        cell += [[0], [self.rel_vocab.safe_pad_token]]
                    sample.append(cell)
            # insert root word with arbitrary fields, anyway it will be masked
            form, cpos, head, deprel = sample[0]
            sample.insert(0, [self.bos, self.bos, [0], deprel])
            yield sample

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        types = (tf.int64, tf.int64), (tf.bool, tf.int64)
        shapes = ([None, None], [None, None]), ([None, None, None], [None, None, None])
        values = (self.form_vocab.safe_pad_token_idx, self.cpos_vocab.safe_pad_token_idx), (
            False, self.rel_vocab.safe_pad_token_idx)
        return types, shapes, values

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None) -> Iterable:
        arc_preds, rel_preds, mask = Y
        sents = []

        for arc_sent, rel_sent, length in zip(arc_preds, rel_preds,
                                              tf.math.count_nonzero(mask, axis=-1)):
            sent = []
            for arc, rel in zip(tolist(arc_sent[1:, 1:]), tolist(rel_sent[1:, 1:])):
                ar = []
                for idx, (a, r) in enumerate(zip(arc, rel)):
                    if a:
                        ar.append((idx + 1, self.rel_vocab.idx_to_token[r]))
                if not ar:
                    # orphan
                    ar.append((0, self.orphan_relation))
                sent.append(ar)
            sents.append(sent)

        return sents

    def XY_to_inputs_outputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]], Y: Union[tf.Tensor, Tuple[tf.Tensor]],
                             gold=False, inputs=None, conll=True) -> Iterable:
        (words, feats, mask), (arc_preds, rel_preds) = X, Y
        xs = inputs
        ys = self.Y_to_outputs((arc_preds, rel_preds, mask))
        sents = []
        for x, y in zip(xs, ys):
            sent = CoNLLSentence()
            for idx, ((form, cpos), pred) in enumerate(zip(x, y)):
                head = [p[0] for p in pred]
                deprel = [p[1] for p in pred]
                if conll:
                    sent.append(CoNLLWord(id=idx + 1, form=form, cpos=cpos, head=head, deprel=deprel))
                else:
                    sent.append([head, deprel])
            sents.append(sent)
        return sents

    def batched_inputs_to_batches(self, corpus, indices, shuffle=False):
        """
        Convert batched inputs to batches of samples

        Parameters
        ----------
        corpus : list
            A list of inputs
        indices : list
            A list of indices, each list belongs to a batch

        Returns
        -------
        None

        Yields
        -------
        tuple
            tuple of tf.Tensor
        """
        raw_batch = [[], [], [], []]
        max_len = len(max([corpus[i] for i in indices], key=len))
        for idx in indices:
            arc = np.zeros((max_len, max_len), dtype=np.bool)
            rel = np.zeros((max_len, max_len), dtype=np.int64)
            for b in raw_batch[:2]:
                b.append([])
            for m, cells in enumerate(corpus[idx]):
                for b, c, v in zip(raw_batch, cells,
                                   [self.form_vocab, self.cpos_vocab]):
                    b[-1].append(v.get_idx_without_add(c))
                for n, r in zip(cells[2], cells[3]):
                    arc[m, n] = True
                    rid = self.rel_vocab.get_idx_without_add(r)
                    if rid is None:
                        logger.warning(f'Relation OOV: {r} not exists in train')
                        continue
                    rel[m, n] = rid
            raw_batch[-2].append(arc)
            raw_batch[-1].append(rel)
        batch = []
        for b, v in zip(raw_batch, [self.form_vocab, self.cpos_vocab]):
            b = tf.keras.preprocessing.sequence.pad_sequences(b, padding='post',
                                                              value=v.safe_pad_token_idx,
                                                              dtype='int64')
            batch.append(b)
        batch += raw_batch[2:]
        assert len(batch) == 4
        yield (batch[0], batch[1]), (batch[2], batch[3])
