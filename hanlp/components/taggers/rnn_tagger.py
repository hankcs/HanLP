# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-20 13:12
import logging

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from hanlp.common.dataset import PadSequenceDataLoader, SortingSampler, TransformableDataset
from hanlp_common.configurable import Configurable
from hanlp.common.transform import EmbeddingNamedTransform
from hanlp.common.vocab import Vocab
from hanlp.components.taggers.rnn.rnntaggingmodel import RNNTaggingModel
from hanlp.components.taggers.tagger import Tagger
from hanlp.datasets.ner.tsv import TSVTaggingDataset
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.layers.embeddings.util import build_word2vec_with_vocab
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, merge_dict


class RNNTagger(Tagger):

    def __init__(self, **kwargs) -> None:
        """An old-school tagger using non-contextualized embeddings and RNNs as context layer.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self.model: RNNTaggingModel = None

    # noinspection PyMethodOverriding
    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion,
                              optimizer,
                              metric,
                              save_dir,
                              logger,
                              patience,
                              **kwargs):
        max_e, max_metric = 0, -1

        criterion = self.build_criterion()
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        scheduler = self.build_scheduler(**merge_dict(self.config, optimizer=optimizer, overwrite=True))
        if not patience:
            patience = epochs
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, ratio_width=ratio_width)
            loss, dev_metric = self.evaluate_dataloader(dev, criterion, logger)
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(dev_metric.score)
                else:
                    scheduler.step(epoch)
            report_patience = f'Patience: {epoch - max_e}/{patience}'
            # save the model if it is the best so far
            if dev_metric > max_metric:
                self.save_weights(save_dir)
                max_e, max_metric = epoch, dev_metric
                report_patience = '[red]Saved[/red] '
            stop = epoch - max_e >= patience
            if stop:
                timer.stop()
            timer.log(f'{report_patience} lr: {optimizer.param_groups[0]["lr"]:.4f}',
                      ratio_percentage=False, newline=True, ratio=False)
            if stop:
                break
        timer.stop()
        if max_e != epoch:
            self.load_weights(save_dir)
        logger.info(f"Max score of dev is {max_metric.score:.2%} at epoch {max_e}")
        logger.info(f"{timer.elapsed_human} elapsed, average time of each epoch is {timer.elapsed_average_human}")

    def build_scheduler(self, optimizer, anneal_factor, anneal_patience, **kwargs):
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer,
                                                         factor=anneal_factor,
                                                         patience=anneal_patience,
                                                         mode='max') if anneal_factor and anneal_patience else None
        return scheduler

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, ratio_width=None,
                       **kwargs):
        self.model.train()
        timer = CountdownTimer(len(trn))
        total_loss = 0
        for idx, batch in enumerate(trn):
            optimizer.zero_grad()
            out, mask = self.feed_batch(batch)
            y = batch['tag_id']
            loss = self.compute_loss(criterion, out, y, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            prediction = self.decode_output(out, mask, batch)
            self.update_metrics(metric, out, y, mask, batch, prediction)
            timer.log(f'loss: {loss / (idx + 1):.4f} {metric}', ratio_percentage=False, logger=logger,
                      ratio_width=ratio_width)
            del loss
            del out
            del mask

    def feed_batch(self, batch):
        x = batch[f'{self.config.token_key}_id']
        out, mask = self.model(x, **batch, batch=batch)
        return out, mask

    # noinspection PyMethodOverriding
    def build_model(self, rnn_input, rnn_hidden, drop, crf, **kwargs) -> torch.nn.Module:
        vocabs = self.vocabs
        token_embed = self._convert_embed()
        if isinstance(token_embed, EmbeddingNamedTransform):
            token_embed = token_embed.output_dim
        elif isinstance(token_embed, Embedding):
            token_embed = token_embed.module(vocabs=vocabs)
        else:
            token_embed = build_word2vec_with_vocab(token_embed, vocabs[self.config.token_key])
        model = RNNTaggingModel(token_embed, rnn_input, rnn_hidden, len(vocabs['tag']), drop, crf)
        return model

    def _convert_embed(self):
        embed = self.config['embed']
        if isinstance(embed, dict):
            self.config['embed'] = embed = Configurable.from_config(embed)
        return embed

    def build_dataloader(self, data, batch_size, shuffle, device, logger=None, **kwargs) -> DataLoader:
        vocabs = self.vocabs
        token_embed = self._convert_embed()
        dataset = data if isinstance(data, TransformableDataset) else self.build_dataset(data, transform=[vocabs])
        if vocabs.mutable:
            # Before building vocabs, let embeddings submit their vocabs, some embeddings will possibly opt out as their
            # transforms are not relevant to vocabs
            if isinstance(token_embed, Embedding):
                transform = token_embed.transform(vocabs=vocabs)
                if transform:
                    dataset.transform.insert(-1, transform)
            self.build_vocabs(dataset, logger)
        if isinstance(token_embed, Embedding):
            # Vocabs built, now add all transforms to the pipeline. Be careful about redundant ones.
            transform = token_embed.transform(vocabs=vocabs)
            if transform and transform not in dataset.transform:
                dataset.transform.insert(-1, transform)
        sampler = SortingSampler([len(sample[self.config.token_key]) for sample in dataset], batch_size,
                                 shuffle=shuffle)
        return PadSequenceDataLoader(dataset,
                                     device=device,
                                     batch_sampler=sampler,
                                     vocabs=vocabs)

    def build_dataset(self, data, transform):
        return TSVTaggingDataset(data, transform)

    def build_vocabs(self, dataset, logger):
        self.vocabs.tag = Vocab(unk_token=None, pad_token=None)
        self.vocabs[self.config.token_key] = Vocab()
        for each in dataset:
            pass
        self.vocabs.lock()
        self.vocabs.summary(logger)

    def fit(self, trn_data, dev_data, save_dir,
            batch_size=50,
            epochs=100,
            embed=100,
            rnn_input=None,
            rnn_hidden=256,
            drop=0.5,
            lr=0.001,
            patience=10,
            crf=True,
            optimizer='adam',
            token_key='token',
            tagging_scheme=None,
            anneal_factor: float = 0.5,
            anneal_patience=2,
            devices=None, logger=None, verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def _id_to_tags(self, ids):
        batch = []
        vocab = self.vocabs['tag'].idx_to_token
        for b in ids:
            batch.append([])
            for i in b:
                batch[-1].append(vocab[i])
        return batch

    def write_output(self, yhat, y, mask, batch, prediction, output):
        pass
