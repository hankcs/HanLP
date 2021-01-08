# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-21 16:26
import os
from typing import Union, List, Callable, Dict

from hanlp_common.constant import NULL
from hanlp.common.dataset import TransformableDataset
import json
from alnlp.metrics import span_utils
from hanlp.utils.io_util import TimingFileIterator, read_tsv_as_sents


class JsonNERDataset(TransformableDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 generate_idx=None, doc_level_offset=True, tagset=None) -> None:
        """A dataset for ``.jsonlines`` format NER corpora.

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
            doc_level_offset: ``True`` to indicate the offsets in ``jsonlines`` are of document level.
            tagset: Optional tagset to prune entities outside of this tagset from datasets.
        """
        self.tagset = tagset
        self.doc_level_offset = doc_level_offset
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        """Load ``.jsonlines`` NER corpus. Samples of this corpus can be found using the following scripts.

        .. highlight:: python
        .. code-block:: python

            import json
            from hanlp_common.document import Document
            from hanlp.datasets.srl.ontonotes5.chinese import ONTONOTES5_CONLL12_CHINESE_DEV
            from hanlp.utils.io_util import get_resource

            with open(get_resource(ONTONOTES5_CONLL12_CHINESE_DEV)) as src:
                for line in src:
                    doc = json.loads(line)
                    print(Document(doc))
                    break

        Args:
            filepath: ``.jsonlines`` NER corpus.
        """
        filename = os.path.basename(filepath)
        reader = TimingFileIterator(filepath)
        num_docs, num_sentences = 0, 0
        for line in reader:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            num_docs += 1
            num_tokens_in_doc = 0
            for sentence, ner in zip(doc['sentences'], doc['ner']):
                if self.doc_level_offset:
                    ner = [(x[0] - num_tokens_in_doc, x[1] - num_tokens_in_doc, x[2]) for x in ner]
                else:
                    ner = [(x[0], x[1], x[2]) for x in ner]
                if self.tagset:
                    ner = [x for x in ner if x[2] in self.tagset]
                    if isinstance(self.tagset, dict):
                        ner = [(x[0], x[1], self.tagset[x[2]]) for x in ner]
                deduplicated_srl = []
                be_set = set()
                for b, e, l in ner:
                    be = (b, e)
                    if be in be_set:
                        continue
                    be_set.add(be)
                    deduplicated_srl.append((b, e, l))
                yield {
                    'token': sentence,
                    'ner': deduplicated_srl
                }
                num_sentences += 1
                num_tokens_in_doc += len(sentence)
            reader.log(
                f'{filename} {num_docs} documents, {num_sentences} sentences [blink][yellow]...[/yellow][/blink]')
        reader.erase()


def convert_conll03_to_json(file_path):
    dataset = []
    num_docs = [0]

    def new_doc():
        doc_key = num_docs[0]
        num_docs[0] += 1
        return {
            'doc_key': doc_key,
            'sentences': [],
            'ner': [],
        }

    doc = new_doc()
    offset = 0
    for cells in read_tsv_as_sents(file_path):
        if cells[0][0] == '-DOCSTART-' and doc['ner']:
            dataset.append(doc)
            doc = new_doc()
            offset = 0
        sentence = [x[0] for x in cells]
        ner = [x[-1] for x in cells]
        ner = span_utils.iobes_tags_to_spans(ner)
        adjusted_ner = []
        for label, (span_start, span_end) in ner:
            adjusted_ner.append([span_start + offset, span_end + offset, label])
        doc['sentences'].append(sentence)
        doc['ner'].append(adjusted_ner)
        offset += len(sentence)
    if doc['ner']:
        dataset.append(doc)
    output_path = os.path.splitext(file_path)[0] + '.json'
    with open(output_path, 'w') as out:
        for each in dataset:
            json.dump(each, out)
            out.write('\n')


def unpack_ner(sample: dict) -> dict:
    ner: list = sample.get('ner', None)
    if ner is not None:
        if ner:
            sample['begin_offset'], sample['end_offset'], sample['label'] = zip(*ner)
        else:
            # It's necessary to create a null label when there is no NER in the sentence for the sake of padding.
            sample['begin_offset'], sample['end_offset'], sample['label'] = [0], [0], [NULL]
    return sample


def prune_ner_tagset(sample: dict, tagset: Union[set, Dict[str, str]]):
    if 'tag' in sample:
        pruned_tag = []
        for tag in sample['tag']:
            cells = tag.split('-', 1)
            if len(cells) == 2:
                role, ner_type = cells
                if ner_type in tagset:
                    if isinstance(tagset, dict):
                        tag = role + '-' + tagset[ner_type]
                else:
                    tag = 'O'
            pruned_tag.append(tag)
        sample['tag'] = pruned_tag
    return sample
