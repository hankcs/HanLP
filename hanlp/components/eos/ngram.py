# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-26 20:19
import logging
from collections import Counter
from typing import Union, List, Callable

import torch
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from hanlp.common.dataset import PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.common.vocab import Vocab
from hanlp.datasets.eos.eos import SentenceBoundaryDetectionDataset
from hanlp.metrics.f1 import F1
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs


class NgramSentenceBoundaryDetectionModel(nn.Module):

    def __init__(self,
                 char_vocab_size,
                 embedding_size=128,
                 rnn_type: str = 'LSTM',
                 rnn_size=256,
                 rnn_layers=1,
                 rnn_bidirectional=False,
                 dropout=0.2,
                 **kwargs
                 ):
        super(NgramSentenceBoundaryDetectionModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=char_vocab_size,
                                  embedding_dim=embedding_size)
        rnn_type = rnn_type.lower()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=rnn_size,
                               num_layers=rnn_layers,
                               dropout=self.dropout if rnn_layers > 1 else 0.0,
                               bidirectional=rnn_bidirectional,
                               batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embdding_size,
                              hidden_size=rnn_size,
                              num_layers=rnn_layers,
                              dropout=self.dropout if rnn_layers > 1 else 0.0,
                              bidirectional=rnn_bidirectional,
                              batch_first=True)
        else:
            raise NotImplementedError(f"'{rnn_type}' has to be one of [LSTM, GRU]")
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.dense = nn.Linear(in_features=rnn_size * (2 if rnn_bidirectional else 1),
                               out_features=1)

    def forward(self, x: torch.Tensor):
        output = self.embed(x)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(output)
        if self.dropout:
            output = self.dropout(output[:, -1, :])
        output = output.squeeze(1)
        output = self.dense(output).squeeze(-1)
        return output


class NgramSentenceBoundaryDetector(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """A sentence boundary detector using ngram as features and LSTM as encoder (:cite:`Schweter:Ahmed:2019`).
        It predicts whether a punctuation marks an ``EOS``.

        .. Note::
            This component won't work on text without the punctuations defined in its config. It's always
            recommended to understand how it works before using it. The predefined punctuations can be listed by the
            following codes.

            >>> print(eos.config.eos_chars)

        Args:
            **kwargs: Passed to config.
        """
        super().__init__(**kwargs)

    def build_optimizer(self, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def build_criterion(self, **kwargs):
        return BCEWithLogitsLoss()

    def build_metric(self, **kwargs):
        return F1()

    def execute_training_loop(self,
                              trn: DataLoader,
                              dev: DataLoader,
                              epochs,
                              criterion,
                              optimizer,
                              metric,
                              save_dir,
                              logger: logging.Logger,
                              devices,
                              **kwargs):
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger)
            if dev:
                self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = metric.score
            if dev_score > best_metric:
                self.save_weights(save_dir)
                best_metric = dev_score
                report += ' [red]saved[/red]'
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)

    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric,
                       logger: logging.Logger,
                       **kwargs):
        self.model.train()
        timer = CountdownTimer(len(trn))
        total_loss = 0
        self.reset_metrics(metric)
        for batch in trn:
            optimizer.zero_grad()
            prediction = self.feed_batch(batch)
            loss = self.compute_loss(prediction, batch, criterion)
            self.update_metrics(batch, prediction, metric)
            loss.backward()
            if self.config.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
            optimizer.step()
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger)
            del loss
        return total_loss / timer.total

    def compute_loss(self, prediction, batch, criterion):
        loss = criterion(prediction, batch['label_id'])
        return loss

    # noinspection PyMethodOverriding
    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric,
                            logger,
                            ratio_width=None,
                            output=False,
                            **kwargs):
        self.model.eval()
        self.reset_metrics(metric)
        timer = CountdownTimer(len(data))
        total_loss = 0
        for batch in data:
            prediction = self.feed_batch(batch)
            self.update_metrics(batch, prediction, metric)
            loss = self.compute_loss(prediction, batch, criterion)
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger,
                      ratio_width=ratio_width)
            del loss
        return total_loss / timer.total, metric

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        model = NgramSentenceBoundaryDetectionModel(**self.config, char_vocab_size=len(self.vocabs.char))
        return model

    def build_dataloader(self, data, batch_size, shuffle, device, logger: logging.Logger, **kwargs) -> DataLoader:
        dataset = SentenceBoundaryDetectionDataset(data, **self.config, transform=[self.vocabs])
        if isinstance(data, str):
            dataset.purge_cache()
        if not self.vocabs:
            self.build_vocabs(dataset, logger)
        return PadSequenceDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, device=device,
                                     pad={'label_id': .0})

    def predict(self, data: Union[str, List[str]], batch_size: int = None, strip=True, **kwargs):
        """Sentence split.

        Args:
            data: A paragraph or a list of paragraphs.
            batch_size: Number of samples per batch.
            strip: Strip out blank characters at the head and tail of each sentence.

        Returns:
            A list of sentences or a list of lists of sentences.
        """
        if not data:
            return []
        self.model.eval()
        flat = isinstance(data, str)
        if flat:
            data = [data]
        samples = []
        eos_chars = self.config.eos_chars
        window_size = self.config.window_size
        for doc_id_, corpus in enumerate(data):
            corpus = list(corpus)
            for i, c in enumerate(corpus):
                if c in eos_chars:
                    window = corpus[max(0, i - window_size): i + window_size + 1]
                    samples.append({'char': window, 'offset_': i, 'doc_id_': doc_id_})
        eos_prediction = [[] for _ in range(len(data))]
        if samples:
            dataloader = self.build_dataloader(samples, **self.config, device=self.device, shuffle=False, logger=None)
            for batch in dataloader:
                logits = self.feed_batch(batch)
                prediction = (logits > 0).tolist()
                for doc_id_, offset_, eos in zip(batch['doc_id_'], batch['offset_'], prediction):
                    if eos:
                        eos_prediction[doc_id_].append(offset_)
        outputs = []
        for corpus, output in zip(data, eos_prediction):
            sents_per_document = []
            prev_offset = 0
            for offset in output:
                offset += 1
                sents_per_document.append(corpus[prev_offset:offset])
                prev_offset = offset
            if prev_offset != len(corpus):
                sents_per_document.append(corpus[prev_offset:])
            if strip:
                sents_per_document = [x.strip() for x in sents_per_document]
            sents_per_document = [x for x in sents_per_document if x]
            outputs.append(sents_per_document)
        if flat:
            outputs = outputs[0]
        return outputs

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            epochs=5,
            append_after_sentence=None,
            eos_chars=None,
            eos_char_min_freq=200,
            eos_char_is_punct=True,
            char_min_freq=None,
            window_size=5,
            batch_size=32,
            lr=0.001,
            grad_norm=None,
            loss_reduction='sum',
            embedding_size=128,
            rnn_type: str = 'LSTM',
            rnn_size=256,
            rnn_layers=1,
            rnn_bidirectional=False,
            dropout=0.2,
            devices=None,
            logger=None,
            seed=None,
            **kwargs
            ):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, dataset: SentenceBoundaryDetectionDataset, logger, **kwargs):
        char_min_freq = self.config.char_min_freq
        if char_min_freq:
            has_cache = dataset.cache is not None
            char_counter = Counter()
            for each in dataset:
                for c in each['char']:
                    char_counter[c] += 1
            self.vocabs.char = vocab = Vocab()
            for c, f in char_counter.items():
                if f >= char_min_freq:
                    vocab.add(c)
            if has_cache:
                dataset.purge_cache()
                for each in dataset:
                    pass
        else:
            self.vocabs.char = Vocab()
            for each in dataset:
                pass
        self.config.eos_chars = dataset.eos_chars
        self.vocabs.lock()
        self.vocabs.summary(logger)

    def reset_metrics(self, metrics):
        metrics.reset()

    def report_metrics(self, loss, metrics):
        return f'loss: {loss:.4f} {metrics}'

    def update_metrics(self, batch: dict, prediction: torch.FloatTensor, metrics):
        def nonzero_offsets(y):
            return set(y.nonzero().squeeze(-1).tolist())

        metrics(nonzero_offsets(prediction > 0), nonzero_offsets(batch['label_id']))

    def feed_batch(self, batch):
        prediction = self.model(batch['char_id'])
        return prediction
