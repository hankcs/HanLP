# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 15:30
from abc import abstractmethod
from collections import Counter
from typing import Union, Tuple, Iterable, Any, Generator

import numpy as np
import tensorflow as tf
from transformers import PreTrainedTokenizer, PretrainedConfig

from hanlp_common.constant import ROOT
from hanlp_common.structure import SerializableDict
from hanlp.common.transform_tf import Transform
from hanlp.common.vocab_tf import VocabTF
from hanlp.components.parsers.alg_tf import tolist, kmeans, randperm, arange
from hanlp.components.parsers.conll import read_conll
from hanlp_common.conll import CoNLLWord, CoNLLUWord, CoNLLSentence
from hanlp.layers.transformers.utils_tf import config_is, adjust_tokens_for_transformers, convert_examples_to_features
from hanlp.utils.log_util import logger
from hanlp.utils.string_util import ispunct
from hanlp_common.util import merge_locals_kwargs


class CoNLLTransform(Transform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=True, n_buckets=32, min_freq=2,
                 use_pos=True, **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.form_vocab: VocabTF = None
        if use_pos:
            self.cpos_vocab: VocabTF = None
        self.rel_vocab: VocabTF = None
        self.puncts: tf.Tensor = None

    @property
    def use_pos(self):
        return self.config.get('use_pos', True)

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
        use_pos = self.use_pos
        conllu = filepath.endswith('.conllu')
        for sent in read_conll(filepath):
            for i, cell in enumerate(sent):
                form = cell[1]
                cpos = cell[3]
                head = cell[6]
                deprel = cell[7]
                # if conllu:
                #     deps = cell[8]
                #     deps = [x.split(':', 1) for x in deps.split('|')]
                #     heads = [int(x[0]) for x in deps if '_' not in x[0] and '.' not in x[0]]
                #     rels = [x[1] for x in deps if '_' not in x[0] and '.' not in x[0]]
                #     if head in heads:
                #         offset = heads.index(head)
                #         if not self.rel_vocab or rels[offset] in self.rel_vocab:
                #             deprel = rels[offset]
                sent[i] = [form, cpos, head, deprel] if use_pos else [form, head, deprel]
            yield sent

    @property
    def bos(self):
        if self.form_vocab.idx_to_token is None:
            return ROOT
        return self.form_vocab.idx_to_token[2]

    def input_is_single_sample(self, input: Any) -> bool:
        if self.use_pos:
            return isinstance(input[0][0], str) if len(input[0]) else False
        else:
            return isinstance(input[0], str) if len(input[0]) else False

    @abstractmethod
    def batched_inputs_to_batches(self, corpus, indices, shuffle):
        pass

    def len_of_sent(self, sent):
        return 1 + len(sent)  # take ROOT into account

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


class CoNLL_DEP_Transform(CoNLLTransform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=True, n_buckets=32,
                 min_freq=2, **kwargs) -> None:
        super().__init__(config, map_x, map_y, lower, n_buckets, min_freq, **kwargs)

    def batched_inputs_to_batches(self, corpus, indices, shuffle):
        """Convert batched inputs to batches of samples

        Args:
          corpus(list): A list of inputs
          indices(list): A list of indices, each list belongs to a batch
          shuffle:

        Returns:


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

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        types = (tf.int64, tf.int64), (tf.int64, tf.int64)
        shapes = ([None, None], [None, None]), ([None, None], [None, None])
        values = (self.form_vocab.safe_pad_token_idx, self.cpos_vocab.safe_pad_token_idx), (
            0, self.rel_vocab.safe_pad_token_idx)
        return types, shapes, values

    def inputs_to_samples(self, inputs, gold=False):
        token_mapping: dict = self.config.get('token_mapping', None)
        use_pos = self.config.get('use_pos', True)
        for sent in inputs:
            sample = []
            for i, cell in enumerate(sent):
                if isinstance(cell, tuple):
                    cell = list(cell)
                elif isinstance(cell, str):
                    cell = [cell]
                if token_mapping:
                    cell[0] = token_mapping.get(cell[0], cell[0])
                if self.config['lower']:
                    cell[0] = cell[0].lower()
                if not gold:
                    cell += [0, self.rel_vocab.safe_pad_token]
                sample.append(cell)
            # insert root word with arbitrary fields, anyway it will be masked
            # form, cpos, head, deprel = sample[0]
            sample.insert(0, [self.bos, self.bos, 0, self.bos] if use_pos else [self.bos, 0, self.bos])
            yield sample

    def XY_to_inputs_outputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]], Y: Union[tf.Tensor, Tuple[tf.Tensor]],
                             gold=False, inputs=None, conll=True, arc_scores=None, rel_scores=None) -> Iterable:
        (words, feats, mask), (arc_preds, rel_preds) = X, Y
        if inputs is None:
            inputs = self.X_to_inputs(X)
        ys = self.Y_to_outputs((arc_preds, rel_preds, mask), inputs=inputs)
        sents = []
        for x, y in zip(inputs, ys):
            sent = CoNLLSentence()
            for idx, (cell, (head, deprel)) in enumerate(zip(x, y)):
                if self.use_pos and not self.config.get('joint_pos', None):
                    form, cpos = cell
                else:
                    form, cpos = cell, None
                if conll:
                    sent.append(
                        CoNLLWord(id=idx + 1, form=form, cpos=cpos, head=head, deprel=deprel) if conll == '.conll'
                        else CoNLLUWord(id=idx + 1, form=form, upos=cpos, head=head, deprel=deprel))
                else:
                    sent.append([head, deprel])
            sents.append(sent)
        return sents

    def fit(self, trn_path: str, **kwargs) -> int:
        use_pos = self.config.use_pos
        self.form_vocab = VocabTF()
        self.form_vocab.add(ROOT)  # make root the 2ed elements while 0th is pad, 1st is unk
        if self.use_pos:
            self.cpos_vocab = VocabTF(pad_token=None, unk_token=None)
        self.rel_vocab = VocabTF(pad_token=None, unk_token=None)
        num_samples = 0
        counter = Counter()
        for sent in self.file_to_samples(trn_path, gold=True):
            num_samples += 1
            for idx, cell in enumerate(sent):
                if use_pos:
                    form, cpos, head, deprel = cell
                else:
                    form, head, deprel = cell
                if idx == 0:
                    root = form
                else:
                    counter[form] += 1
                if use_pos:
                    self.cpos_vocab.add(cpos)
                self.rel_vocab.add(deprel)

        for token in [token for token, freq in counter.items() if freq >= self.config.min_freq]:
            self.form_vocab.add(token)
        return num_samples

    @property
    def root_rel_idx(self):
        root_rel_idx = self.config.get('root_rel_idx', None)
        if root_rel_idx is None:
            for idx, rel in enumerate(self.rel_vocab.idx_to_token):
                if 'root' in rel.lower() and rel != self.bos:
                    self.config['root_rel_idx'] = root_rel_idx = idx
                    break
        return root_rel_idx

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None) -> Iterable:
        arc_preds, rel_preds, mask = Y
        sents = []

        for arc_sent, rel_sent, length in zip(arc_preds, rel_preds,
                                              tf.math.count_nonzero(mask, axis=-1)):
            arcs = tolist(arc_sent)[1:length + 1]
            rels = tolist(rel_sent)[1:length + 1]
            sents.append([(a, self.rel_vocab.idx_to_token[r]) for a, r in zip(arcs, rels)])

        return sents


class CoNLL_Transformer_Transform(CoNLL_DEP_Transform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True,
                 lower=True, n_buckets=32, min_freq=0, max_seq_length=256, use_pos=False,
                 mask_p=None, graph=False, topk=None,
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.tokenizer: PreTrainedTokenizer = None
        self.transformer_config: PretrainedConfig = None
        if graph:
            self.orphan_relation = ROOT

    def lock_vocabs(self):
        super().lock_vocabs()
        if self.graph:
            CoNLL_SDP_Transform._find_orphan_relation(self)

    def fit(self, trn_path: str, **kwargs) -> int:
        if self.config.get('joint_pos', None):
            self.config.use_pos = True
        if self.graph:
            # noinspection PyCallByClass
            num = CoNLL_SDP_Transform.fit(self, trn_path, **kwargs)
        else:
            num = super().fit(trn_path, **kwargs)
        if self.config.get('topk', None):
            counter = Counter()
            for sent in self.file_to_samples(trn_path, gold=True):
                for idx, cell in enumerate(sent):
                    form, head, deprel = cell
                    counter[form] += 1
            self.topk_vocab = VocabTF()
            for k, v in counter.most_common(self.config.topk):
                self.topk_vocab.add(k)
        return num

    def inputs_to_samples(self, inputs, gold=False):
        if self.graph:
            yield from CoNLL_SDP_Transform.inputs_to_samples(self, inputs, gold)
        else:
            yield from super().inputs_to_samples(inputs, gold)

    def file_to_inputs(self, filepath: str, gold=True):
        if self.graph:
            yield from CoNLL_SDP_Transform.file_to_inputs(self, filepath, gold)
        else:
            yield from super().file_to_inputs(filepath, gold)

    @property
    def mask_p(self) -> float:
        return self.config.get('mask_p', None)

    @property
    def graph(self):
        return self.config.get('graph', None)

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        mask_p = self.mask_p
        types = (tf.int64, (tf.int64, tf.int64, tf.int64)), (tf.bool if self.graph else tf.int64, tf.int64, tf.int64) if mask_p else (
            tf.bool if self.graph else tf.int64, tf.int64)
        if self.graph:
            shapes = ([None, None], ([None, None], [None, None], [None, None])), (
                [None, None, None], [None, None, None], [None, None]) if mask_p else (
                [None, None, None], [None, None, None])
        else:
            shapes = ([None, None], ([None, None], [None, None], [None, None])), (
                [None, None], [None, None], [None, None]) if mask_p else ([None, None], [None, None])

        values = (self.form_vocab.safe_pad_token_idx, (0, 0, 0)), \
                 (0, self.rel_vocab.safe_pad_token_idx, 0) if mask_p else (0, self.rel_vocab.safe_pad_token_idx)
        types_shapes_values = types, shapes, values
        if self.use_pos:
            types_shapes_values = [((shapes[0][0], shapes[0][1] + (shapes[0][0],)), shapes[1]) for shapes in
                                   types_shapes_values]
        return types_shapes_values

    def X_to_inputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]]) -> Iterable:
        form_batch, feat, prefix_mask = X
        sents = []

        for form_sent, length in zip(form_batch, tf.math.count_nonzero(prefix_mask, axis=-1)):
            forms = tolist(form_sent)[1:length + 1]
            sents.append([self.form_vocab.idx_to_token[f] for f in forms])

        return sents

    def batched_inputs_to_batches(self, corpus, indices, shuffle):
        use_pos = self.use_pos
        if use_pos:
            raw_batch = [[], [], [], []]
        else:
            raw_batch = [[], [], []]
        if self.graph:
            max_len = len(max([corpus[i] for i in indices], key=len))
            for idx in indices:
                arc = np.zeros((max_len, max_len), dtype=np.bool)
                rel = np.zeros((max_len, max_len), dtype=np.int64)
                for b in raw_batch[:2 if use_pos else 1]:
                    b.append([])
                for m, cells in enumerate(corpus[idx]):
                    if use_pos:
                        for b, c, v in zip(raw_batch, cells, [None, self.cpos_vocab]):
                            b[-1].append(v.get_idx_without_add(c) if v else c)
                    else:
                        for b, c, v in zip(raw_batch, cells, [None]):
                            b[-1].append(c)
                    for n, r in zip(cells[-2], cells[-1]):
                        arc[m, n] = True
                        rid = self.rel_vocab.get_idx_without_add(r)
                        if rid is None:
                            logger.warning(f'Relation OOV: {r} not exists in train')
                            continue
                        rel[m, n] = rid
                raw_batch[-2].append(arc)
                raw_batch[-1].append(rel)
        else:
            for idx in indices:
                for s in raw_batch:
                    s.append([])
                for cells in corpus[idx]:
                    if use_pos:
                        for s, c, v in zip(raw_batch, cells, [None, self.cpos_vocab, None, self.rel_vocab]):
                            s[-1].append(v.get_idx_without_add(c) if v else c)
                    else:
                        for s, c, v in zip(raw_batch, cells, [None, None, self.rel_vocab]):
                            s[-1].append(v.get_idx_without_add(c) if v else c)

        # Transformer tokenizing
        config = self.transformer_config
        tokenizer = self.tokenizer
        xlnet = config_is(config, 'xlnet')
        roberta = config_is(config, 'roberta')
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        max_seq_length = self.config.max_seq_length
        batch_forms = []
        batch_input_ids = []
        batch_input_mask = []
        batch_prefix_offset = []
        mask_p = self.mask_p
        if mask_p:
            batch_masked_offsets = []
            mask_token_id = tokenizer.mask_token_id
        for sent_idx, sent in enumerate(raw_batch[0]):
            batch_forms.append([self.form_vocab.get_idx_without_add(token) for token in sent])
            sent = adjust_tokens_for_transformers(sent)
            sent = sent[1:]  # remove <root> use [CLS] instead
            pad_label_idx = self.form_vocab.pad_idx
            input_ids, input_mask, segment_ids, prefix_mask = \
                convert_examples_to_features(sent,
                                             max_seq_length,
                                             tokenizer,
                                             cls_token_at_end=xlnet,
                                             # xlnet has a cls token at the end
                                             cls_token=cls_token,
                                             cls_token_segment_id=2 if xlnet else 0,
                                             sep_token=sep_token,
                                             sep_token_extra=roberta,
                                             # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                             pad_on_left=xlnet,
                                             # pad on the left for xlnet
                                             pad_token_id=pad_token_id,
                                             pad_token_segment_id=4 if xlnet else 0,
                                             pad_token_label_id=pad_label_idx,
                                             do_padding=False)
            num_masks = sum(prefix_mask)
            # assert len(sent) == num_masks  # each token has a True subtoken
            if num_masks < len(sent):  # long sent gets truncated, +1 for root
                batch_forms[-1] = batch_forms[-1][:num_masks + 1]  # form
                raw_batch[-1][sent_idx] = raw_batch[-1][sent_idx][:num_masks + 1]  # head
                raw_batch[-2][sent_idx] = raw_batch[-2][sent_idx][:num_masks + 1]  # rel
                raw_batch[-3][sent_idx] = raw_batch[-3][sent_idx][:num_masks + 1]  # pos
            prefix_mask[0] = True  # <root> is now [CLS]
            prefix_offset = [idx for idx, m in enumerate(prefix_mask) if m]
            batch_input_ids.append(input_ids)
            batch_input_mask.append(input_mask)
            batch_prefix_offset.append(prefix_offset)
            if mask_p:
                if shuffle:
                    size = int(np.ceil(mask_p * len(prefix_offset[1:])))  # never mask [CLS]
                    mask_offsets = np.random.choice(np.arange(1, len(prefix_offset)), size, replace=False)
                    for offset in sorted(mask_offsets):
                        assert 0 < offset < len(input_ids)
                        # mask_word = raw_batch[0][sent_idx][offset]
                        # mask_prefix = tokenizer.convert_ids_to_tokens([input_ids[prefix_offset[offset]]])[0]
                        # assert mask_word.startswith(mask_prefix) or mask_prefix.startswith(
                        #     mask_word) or mask_prefix == "'", \
                        #     f'word {mask_word} prefix {mask_prefix} not match'  # could vs couldn
                        # mask_offsets.append(input_ids[offset]) # subword token
                        # mask_offsets.append(offset)  # form token
                        input_ids[prefix_offset[offset]] = mask_token_id  # mask prefix
                        # whole word masking, mask the rest of the word
                        for i in range(prefix_offset[offset] + 1, len(input_ids) - 1):
                            if prefix_mask[i]:
                                break
                            input_ids[i] = mask_token_id

                    batch_masked_offsets.append(sorted(mask_offsets))
                else:
                    batch_masked_offsets.append([0])  # No masking in prediction

        batch_forms = tf.keras.preprocessing.sequence.pad_sequences(batch_forms, padding='post',
                                                                    value=self.form_vocab.safe_pad_token_idx,
                                                                    dtype='int64')
        batch_input_ids = tf.keras.preprocessing.sequence.pad_sequences(batch_input_ids, padding='post',
                                                                        value=pad_token_id,
                                                                        dtype='int64')
        batch_input_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_input_mask, padding='post',
                                                                         value=0,
                                                                         dtype='int64')
        batch_prefix_offset = tf.keras.preprocessing.sequence.pad_sequences(batch_prefix_offset, padding='post',
                                                                            value=0,
                                                                            dtype='int64')
        batch_heads = tf.keras.preprocessing.sequence.pad_sequences(raw_batch[-2], padding='post',
                                                                    value=0,
                                                                    dtype='int64')
        batch_rels = tf.keras.preprocessing.sequence.pad_sequences(raw_batch[-1], padding='post',
                                                                   value=self.rel_vocab.safe_pad_token_idx,
                                                                   dtype='int64')
        if mask_p:
            batch_masked_offsets = tf.keras.preprocessing.sequence.pad_sequences(batch_masked_offsets, padding='post',
                                                                                 value=pad_token_id,
                                                                                 dtype='int64')
        feats = (tf.constant(batch_input_ids, dtype='int64'), tf.constant(batch_input_mask, dtype='int64'),
                 tf.constant(batch_prefix_offset))
        if use_pos:
            batch_pos = tf.keras.preprocessing.sequence.pad_sequences(raw_batch[1], padding='post',
                                                                      value=self.cpos_vocab.safe_pad_token_idx,
                                                                      dtype='int64')
            feats += (batch_pos,)
        yield (batch_forms, feats), \
              (batch_heads, batch_rels, batch_masked_offsets) if mask_p else (batch_heads, batch_rels)

    def len_of_sent(self, sent):
        # Transformer tokenizing
        config = self.transformer_config
        tokenizer = self.tokenizer
        xlnet = config_is(config, 'xlnet')
        roberta = config_is(config, 'roberta')
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        max_seq_length = self.config.max_seq_length
        sent = sent[1:]  # remove <root> use [CLS] instead
        pad_label_idx = self.form_vocab.pad_idx
        sent = [x[0] for x in sent]
        sent = adjust_tokens_for_transformers(sent)
        input_ids, input_mask, segment_ids, prefix_mask = \
            convert_examples_to_features(sent,
                                         max_seq_length,
                                         tokenizer,
                                         cls_token_at_end=xlnet,
                                         # xlnet has a cls token at the end
                                         cls_token=cls_token,
                                         cls_token_segment_id=2 if xlnet else 0,
                                         sep_token=sep_token,
                                         sep_token_extra=roberta,
                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                         pad_on_left=xlnet,
                                         # pad on the left for xlnet
                                         pad_token_id=pad_token_id,
                                         pad_token_segment_id=4 if xlnet else 0,
                                         pad_token_label_id=pad_label_idx,
                                         do_padding=False)
        return len(input_ids)

    def samples_to_dataset(self, samples: Generator, map_x=None, map_y=None, batch_size=5000, shuffle=None, repeat=None,
                           drop_remainder=False, prefetch=1, cache=True) -> tf.data.Dataset:
        if shuffle:
            return CoNLL_DEP_Transform.samples_to_dataset(self, samples, map_x, map_y, batch_size, shuffle, repeat,
                                                          drop_remainder, prefetch, cache)

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

        # debug for transformer
        # next(generator())
        return Transform.samples_to_dataset(self, generator, False, False, 0, False, repeat, drop_remainder, prefetch,
                                            cache)

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None) -> Iterable:
        if self.graph:
            ys = CoNLL_SDP_Transform.Y_to_outputs(self, Y, gold, inputs, X)
            ys = [[([t[0] for t in l], [t[1] for t in l]) for l in y] for y in ys]
            return ys
        return super().Y_to_outputs(Y, gold, inputs, X)


class CoNLL_SDP_Transform(CoNLLTransform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=True, n_buckets=32, min_freq=2,
                 use_pos=True, **kwargs) -> None:
        super().__init__(config, map_x, map_y, lower, n_buckets, min_freq, use_pos, **kwargs)
        self.orphan_relation = ROOT

    def lock_vocabs(self):
        super().lock_vocabs()
        # heuristic to find the orphan relation
        self._find_orphan_relation()

    def _find_orphan_relation(self):
        for rel in self.rel_vocab.idx_to_token:
            if 'root' in rel.lower():
                self.orphan_relation = rel
                break

    def file_to_inputs(self, filepath: str, gold=True):
        assert gold, 'only support gold file for now'
        use_pos = self.use_pos
        conllu = filepath.endswith('.conllu')
        enhanced_only = self.config.get('enhanced_only', None)
        for i, sent in enumerate(read_conll(filepath)):
            parsed_sent = []
            if conllu:
                for cell in sent:
                    ID = cell[0]
                    form = cell[1]
                    cpos = cell[3]
                    head = cell[6]
                    deprel = cell[7]
                    deps = cell[8]
                    deps = [x.split(':', 1) for x in deps.split('|')]
                    heads = [int(x[0]) for x in deps if x[0].isdigit()]
                    rels = [x[1] for x in deps if x[0].isdigit()]
                    if enhanced_only:
                        if head in heads:
                            offset = heads.index(head)
                            heads.pop(offset)
                            rels.pop(offset)
                    else:
                        if head not in heads:
                            heads.append(head)
                            rels.append(deprel)
                    parsed_sent.append([form, cpos, heads, rels] if use_pos else [form, heads, rels])
            else:
                prev_cells = None
                heads = []
                rels = []
                for j, cell in enumerate(sent):
                    ID = cell[0]
                    form = cell[1]
                    cpos = cell[3]
                    head = cell[6]
                    deprel = cell[7]
                    if prev_cells and ID != prev_cells[0]:  # found end of token
                        parsed_sent.append(
                            [prev_cells[1], prev_cells[2], heads, rels] if use_pos else [prev_cells[1], heads, rels])
                        heads = []
                        rels = []
                    heads.append(head)
                    rels.append(deprel)
                    prev_cells = [ID, form, cpos, head, deprel] if use_pos else [ID, form, head, deprel]
                parsed_sent.append(
                    [prev_cells[1], prev_cells[2], heads, rels] if use_pos else [prev_cells[1], heads, rels])
            yield parsed_sent

    def fit(self, trn_path: str, **kwargs) -> int:
        self.form_vocab = VocabTF()
        self.form_vocab.add(ROOT)  # make root the 2ed elements while 0th is pad, 1st is unk
        if self.use_pos:
            self.cpos_vocab = VocabTF(pad_token=None, unk_token=None)
        self.rel_vocab = VocabTF(pad_token=None, unk_token=None)
        num_samples = 0
        counter = Counter()
        for sent in self.file_to_samples(trn_path, gold=True):
            num_samples += 1
            for idx, cell in enumerate(sent):
                if len(cell) == 4:
                    form, cpos, head, deprel = cell
                elif len(cell) == 3:
                    if self.use_pos:
                        form, cpos = cell[0]
                    else:
                        form = cell[0]
                    head, deprel = cell[1:]
                else:
                    raise ValueError('Unknown data arrangement')
                if idx == 0:
                    root = form
                else:
                    counter[form] += 1
                if self.use_pos:
                    self.cpos_vocab.add(cpos)
                self.rel_vocab.update(deprel)

        for token in [token for token, freq in counter.items() if freq >= self.config.min_freq]:
            self.form_vocab.add(token)
        return num_samples

    def inputs_to_samples(self, inputs, gold=False):
        use_pos = self.use_pos
        for sent in inputs:
            sample = []
            for i, cell in enumerate(sent):
                if isinstance(cell, tuple):
                    cell = list(cell)
                elif isinstance(cell, str):
                    cell = [cell]
                if self.config['lower']:
                    cell[0] = cell[0].lower()
                if not gold:
                    cell += [[0], [self.rel_vocab.safe_pad_token]]
                sample.append(cell)
            # insert root word with arbitrary fields, anyway it will be masked
            if use_pos:
                form, cpos, head, deprel = sample[0]
                sample.insert(0, [self.bos, self.bos, [0], deprel])
            else:
                form, head, deprel = sample[0]
                sample.insert(0, [self.bos, [0], deprel])
            yield sample

    def batched_inputs_to_batches(self, corpus, indices, shuffle):
        use_pos = self.use_pos
        raw_batch = [[], [], [], []] if use_pos else [[], [], []]
        max_len = len(max([corpus[i] for i in indices], key=len))
        for idx in indices:
            arc = np.zeros((max_len, max_len), dtype=np.bool)
            rel = np.zeros((max_len, max_len), dtype=np.int64)
            for b in raw_batch[:2]:
                b.append([])
            for m, cells in enumerate(corpus[idx]):
                if use_pos:
                    for b, c, v in zip(raw_batch, cells,
                                       [self.form_vocab, self.cpos_vocab]):
                        b[-1].append(v.get_idx_without_add(c))
                else:
                    for b, c, v in zip(raw_batch, cells,
                                       [self.form_vocab]):
                        b[-1].append(v.get_idx_without_add(c))
                for n, r in zip(cells[-2], cells[-1]):
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
