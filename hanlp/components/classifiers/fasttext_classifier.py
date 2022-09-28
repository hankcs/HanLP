# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-09-28 13:31
import os
import sys
from typing import List, Union

import fasttext
from fasttext.FastText import _FastText

import hanlp
from hanlp.common.component import Component
from hanlp.utils.io_util import get_resource, stdout_redirected
from hanlp_common.io import load_json
from hanlp_common.reflection import classpath_of
from hanlp_common.structure import SerializableDict


class FastTextClassifier(Component):

    def __init__(self) -> None:
        super().__init__()
        self._model: _FastText = None
        self.config = SerializableDict({
            'classpath': classpath_of(self),
            'hanlp_version': hanlp.__version__,
        })

    def load(self, save_dir, model_path=None, **kwargs):
        config_path = os.path.join(save_dir, 'config.json')
        if os.path.isfile(config_path):
            self.config: dict = load_json(config_path)
            model_path = self.config.get('model_path', model_path)
        else:
            model_path = model_path or save_dir
            self.config['model_path'] = model_path
        filepath = get_resource(model_path)
        with stdout_redirected(to=os.devnull, stdout=sys.stderr):
            self._model = fasttext.load_model(filepath)

    def predict(self, text: Union[str, List[str]], topk=False, prob=False, max_len=None, **kwargs):
        """
        Classify text.

        Args:
            text: A document or a list of documents.
            topk: ``True`` or ``int`` to return the top-k labels.
            prob: Return also probabilities.
            max_len: Strip long document into ``max_len`` characters for faster prediction.
            **kwargs: Not used

        Returns:
            Classification results.
        """
        num_labels = len(self._model.get_labels())
        flat = isinstance(text, str)
        if flat:
            text = [text]
        if not isinstance(topk, list):
            topk = [topk] * len(text)
        if not isinstance(prob, list):
            prob = [prob] * len(text)
        if max_len:
            text = [x[:max_len] for x in text]
        text = [x.replace('\n', ' ') for x in text]
        batch_labels, batch_probs = self._model.predict(text, k=num_labels)
        results = []
        for labels, probs, k, p in zip(batch_labels, batch_probs, topk, prob):
            labels = [self._strip_prefix(x) for x in labels]
            if k is False:
                labels = labels[0]
            elif k is True:
                pass
            elif k:
                labels = labels[:k]
            if p:
                probs = probs.tolist()
                if k is False:
                    result = labels, probs[0]
                else:
                    result = dict(zip(labels, probs))
            else:
                result = labels
            results.append(result)
        if flat:
            results = results[0]
        return results

    @property
    def labels(self):
        return [self._strip_prefix(x) for x in self._model.get_labels()]

    @staticmethod
    def _strip_prefix(label: str):
        return label[len('__label__'):]
