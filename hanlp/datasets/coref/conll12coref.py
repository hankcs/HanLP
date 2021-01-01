# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-04 15:33
import collections
import os
from typing import Union, List, Callable, DefaultDict, Tuple, Optional, Iterator

from alnlp.data.ontonotes import Ontonotes as _Ontonotes, OntonotesSentence
from alnlp.data.util import make_coref_instance

from hanlp.common.dataset import TransformableDataset
from hanlp.utils.io_util import TimingFileIterator


class Ontonotes(_Ontonotes):
    def dataset_document_iterator(self, file_path: str) -> Iterator[List[OntonotesSentence]]:
        """An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.

        Args:
          file_path: str: 

        Returns:

        """
        open_file = TimingFileIterator(file_path)
        conll_rows = []
        document: List[OntonotesSentence] = []
        for line in open_file:
            open_file.log(f'Loading {os.path.basename(file_path)}')
            line = line.strip()
            if line != "" and not line.startswith("#"):
                # Non-empty line. Collect the annotation.
                conll_rows.append(line)
            else:
                if conll_rows:
                    document.append(self._conll_rows_to_sentence(conll_rows))
                    conll_rows = []
            if line.startswith("#end document"):
                yield document
                document = []
        open_file.erase()
        if document:
            # Collect any stragglers or files which might not
            # have the '#end document' format for the end of the file.
            yield document


class CONLL12CorefDataset(TransformableDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 max_span_width=10, max_sentences=None, remove_singleton_clusters=False) -> None:
        self.remove_singleton_clusters = remove_singleton_clusters
        self.max_sentences = max_sentences
        self.max_span_width = max_span_width
        super().__init__(data, transform, cache)

    def load_file(self, filepath: str):
        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(filepath):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens, end + total_tokens))
                total_tokens += len(sentence.words)

            yield self.text_to_instance([s.words for s in sentences], list(clusters.values()))

    def text_to_instance(
            self,  # type: ignore
            sentences: List[List[str]],
            gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> dict:
        return make_coref_instance(
            sentences,
            self.max_span_width,
            gold_clusters,
            self.max_sentences,
            self.remove_singleton_clusters,
        )
