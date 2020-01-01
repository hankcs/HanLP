# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-25 00:19
import unicodedata
from typing import List, Dict

import tensorflow as tf


def format_metrics(metrics: List[tf.keras.metrics.Metric]):
    return ' - '.join(f'{m.name}: {m.result():.4f}' for m in metrics)


def format_scores(results: Dict[str, float]) -> str:
    return ' - '.join(f'{k}: {v:.4f}' for (k, v) in results.items())


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


