from typing import Dict

import torch
import torch.nn.functional as F
from alnlp.modules import util
from alnlp.modules.time_distributed import TimeDistributed
from torch import nn

from ...parsers.biaffine.biaffine import Biaffine


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


class BiaffineNamedEntityRecognitionModel(nn.Module):

    def __init__(self, config, embed: torch.nn.Module, context_layer: torch.nn.Module, label_space_size):
        super(BiaffineNamedEntityRecognitionModel, self).__init__()
        self.config = config
        self.lexical_dropout = float(self.config.lexical_dropout)
        self.label_space_size = label_space_size

        # Initialize layers and parameters
        self.word_embedding_dim = embed.get_output_dim()  # get the embedding dim
        self.embed = embed
        # Initialize context layer
        self.context_layer = context_layer
        context_layer_output_dim = context_layer.get_output_dim()

        self.decoder = BiaffineNamedEntityRecognitionDecoder(context_layer_output_dim, config.ffnn_size,
                                                             label_space_size, config.loss_reduction)

    def forward(self,
                batch: Dict[str, torch.Tensor]
                ):
        keys = 'token_length', 'begin_offset', 'end_offset', 'label_id'
        sent_lengths, gold_starts, gold_ends, gold_labels = [batch.get(k, None) for k in keys]
        masks = util.lengths_to_mask(sent_lengths)
        num_sentences, max_sent_length = masks.size()
        raw_embeddings = self.embed(batch, mask=masks)

        raw_embeddings = F.dropout(raw_embeddings, self.lexical_dropout, self.training)

        contextualized_embeddings = self.context_layer(raw_embeddings, masks)
        return self.decoder.decode(contextualized_embeddings, gold_starts, gold_ends, gold_labels, masks,
                                   max_sent_length,
                                   num_sentences, sent_lengths)


class BiaffineNamedEntityRecognitionDecoder(nn.Module):
    def __init__(self, hidden_size, ffnn_size, label_space_size, loss_reduction='sum') -> None:
        """An implementation of the biaffine decoder in "Named Entity Recognition as Dependency Parsing"
        (:cite:`yu-etal-2020-named`).

        Args:
            hidden_size: Size of hidden states.
            ffnn_size: Feedforward size for MLPs extracting the head/tail representations.
            label_space_size: Size of tag set.
            loss_reduction: The loss reduction used in aggregating losses.
        """
        super().__init__()
        self.loss_reduction = loss_reduction

        # MLPs
        def new_mlp():
            return TimeDistributed(nn.Linear(hidden_size, ffnn_size))

        self.start_mlp = new_mlp()
        self.end_mlp = new_mlp()
        self.biaffine = Biaffine(ffnn_size, label_space_size)

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        keys = 'token_length', 'begin_offset', 'end_offset', 'label_id'
        sent_lengths, gold_starts, gold_ends, gold_labels = [batch.get(k, None) for k in keys]
        if mask is None:
            mask = util.lengths_to_mask(sent_lengths)
        num_sentences, max_sent_length = mask.size()
        return self.decode(contextualized_embeddings, gold_starts, gold_ends, gold_labels, mask,
                           max_sent_length,
                           num_sentences, sent_lengths)

    def get_dense_span_labels(self, span_starts, span_ends, span_labels, max_sentence_length):
        num_sentences, max_spans_num = span_starts.size()

        sentence_indices = torch.arange(0, num_sentences, device=span_starts.device).unsqueeze(1).expand(-1,
                                                                                                         max_spans_num)

        sparse_indices = torch.cat([sentence_indices.unsqueeze(2), span_starts.unsqueeze(2), span_ends.unsqueeze(2)],
                                   dim=2)
        rank = 3
        dense_labels = torch.sparse.LongTensor(sparse_indices.view(num_sentences * max_spans_num, rank).t(),
                                               span_labels.view(-1),
                                               torch.Size([num_sentences] + [max_sentence_length] * (rank - 1))) \
            .to_dense()
        return dense_labels

    def decode(self, contextualized_embeddings, gold_starts, gold_ends, gold_labels, masks, max_sent_length,
               num_sentences, sent_lengths):
        # Apply MLPs to starts and ends, [num_sentences, max_sentences_length,emb]
        candidate_starts_emb = self.start_mlp(contextualized_embeddings)
        candidate_ends_emb = self.end_mlp(contextualized_embeddings)
        candidate_ner_scores = self.biaffine(candidate_starts_emb, candidate_ends_emb).permute([0, 2, 3, 1])

        """generate candidate spans with argument pruning"""
        # Generate masks
        candidate_scores_mask = masks.unsqueeze(1) & masks.unsqueeze(2)
        device = sent_lengths.device
        sentence_ends_leq_starts = (
            ~util.lengths_to_mask(torch.arange(max_sent_length, device=device), max_sent_length)) \
            .unsqueeze_(0).expand(num_sentences, -1, -1)
        candidate_scores_mask &= sentence_ends_leq_starts
        candidate_ner_scores = candidate_ner_scores[candidate_scores_mask]
        predict_dict = {
            "candidate_ner_scores": candidate_ner_scores,

        }
        if gold_starts is not None:
            gold_ner_labels = self.get_dense_span_labels(gold_starts, gold_ends, gold_labels, max_sent_length)
            loss = torch.nn.functional.cross_entropy(candidate_ner_scores,
                                                     gold_ner_labels[candidate_scores_mask],
                                                     reduction=self.loss_reduction)
            predict_dict['loss'] = loss
        return predict_dict
