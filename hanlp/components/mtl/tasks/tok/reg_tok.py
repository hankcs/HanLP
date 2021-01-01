# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-02 16:51
import logging
from typing import Union, List, Dict, Any, Iterable, Tuple

import torch
from alnlp.modules import util
from torch import Tensor
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import FieldLength, TransformList
from hanlp.components.mtl.tasks import Task
from hanlp.datasets.tokenization.txt import TextTokenizingDataset
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.layers.transformers.pt_imports import PreTrainedTokenizer
from hanlp.metrics.chunking.binary_chunking_f1 import BinaryChunkingF1
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp_common.util import merge_locals_kwargs


def generate_token_span_tuple(sample: dict):
    prefix_mask = sample.get('text_prefix_mask', None)
    if prefix_mask:
        sample['span_tuple'] = spans = []
        previous_prefix = 0
        prefix_mask_ = prefix_mask[1:-1]
        for i, mask in enumerate(prefix_mask_):
            if i and mask:
                spans.append((previous_prefix, i))
                previous_prefix = i
        spans.append((previous_prefix, len(prefix_mask_)))
    return sample


class RegressionTokenizingDecoder(torch.nn.Linear):

    def __init__(self, in_features: int, out_features: int = 1, bias: bool = ...) -> None:
        super().__init__(in_features, out_features, bias)

    # noinspection PyMethodOverriding
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return super().forward(input[:, 1:-1, :]).squeeze_(-1)


class RegressionTokenization(Task):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=True, lr=1e-3, separate_optimizer=False, delimiter=None,
                 max_seq_len=None, sent_delimiter=None) -> None:
        super().__init__(**merge_locals_kwargs(locals()))

    def build_criterion(self, **kwargs):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')

    def build_metric(self, **kwargs):
        return BinaryChunkingF1()

    # noinspection PyMethodOverriding
    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return RegressionTokenizingDecoder(encoder_size)

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        pass

    def build_dataloader(self,
                         data,
                         transform: TransformList = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         tokenizer: PreTrainedTokenizer = None,
                         **kwargs) -> DataLoader:
        assert tokenizer
        dataset = TextTokenizingDataset(data, cache=isinstance(data, str), delimiter=self.config.sent_delimiter,
                                        generate_idx=isinstance(data, list),
                                        max_seq_len=self.config.max_seq_len,
                                        sent_delimiter=self.config.sent_delimiter,
                                        transform=[
                                            TransformerSequenceTokenizer(tokenizer,
                                                                         'text',
                                                                         ret_prefix_mask=True,
                                                                         ret_subtokens=True,
                                                                         ),
                                            FieldLength('text_input_ids', 'text_input_ids_length', delta=-2),
                                            generate_token_span_tuple])
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset, 'text_input_ids', 'text'),
                                                     shuffle=training),
            device=device,
            dataset=dataset)

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      batch: Dict[str, Any], **kwargs) -> List[Tuple[int, int]]:
        spans = BinaryChunkingF1.decode_spans(output > 0, batch['text_input_ids_length'])
        return spans

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: List[Tuple[int, int]], metric: BinaryChunkingF1):
        metric.update(prediction, batch['span_tuple'])

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion):
        mask = util.lengths_to_mask(batch['text_input_ids_length'])
        return criterion(output[mask], batch['text_prefix_mask'][:, 1:-1][mask].to(torch.float))
