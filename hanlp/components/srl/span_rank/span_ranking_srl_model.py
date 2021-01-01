from typing import Dict

from alnlp.modules.feedforward import FeedForward
from alnlp.modules.time_distributed import TimeDistributed

from .highway_variational_lstm import *
import torch
from alnlp.modules import util

from ...parsers.biaffine.biaffine import Biaffine


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


class SpanRankingSRLDecoder(nn.Module):

    def __init__(self, context_layer_output_dim, label_space_size, config) -> None:
        super().__init__()
        self.config = config
        self.label_space_size = label_space_size
        self.dropout = float(config.dropout)
        self.use_gold_predicates = config.use_gold_predicates
        # span width feature embedding
        self.span_width_embedding = nn.Embedding(self.config.max_arg_width, self.config.span_width_feature_size)
        # self.context_projective_layer = nn.Linear(2 * self.lstm_hidden_size, self.config.num_attention_heads)
        # span scores
        self.span_emb_size = 3 * context_layer_output_dim + self.config.span_width_feature_size
        self.arg_unary_score_layers = nn.ModuleList([nn.Linear(self.span_emb_size, self.config.ffnn_size) if i == 0
                                                     else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
                                                     in range(self.config.ffnn_depth)])  # [,150]
        self.arg_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.arg_unary_score_projection = nn.Linear(self.config.ffnn_size, 1)
        # predicate scores
        self.pred_unary_score_layers = nn.ModuleList(
            [nn.Linear(context_layer_output_dim, self.config.ffnn_size) if i == 0
             else nn.Linear(self.config.ffnn_size, self.config.ffnn_size) for i
             in range(self.config.ffnn_depth)])  # [,150]
        self.pred_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.pred_unary_score_projection = nn.Linear(self.config.ffnn_size, 1)
        # srl scores
        self.srl_unary_score_input_size = self.span_emb_size + context_layer_output_dim
        self.srl_unary_score_layers = nn.ModuleList([nn.Linear(self.srl_unary_score_input_size, self.config.ffnn_size)
                                                     if i == 0 else nn.Linear(self.config.ffnn_size,
                                                                              self.config.ffnn_size)
                                                     for i in range(self.config.ffnn_depth)])
        self.srl_dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.config.ffnn_depth)])
        self.srl_unary_score_projection = nn.Linear(self.config.ffnn_size, self.label_space_size - 1)
        if config.use_biaffine:
            self.predicate_scale = TimeDistributed(FeedForward(context_layer_output_dim, 1, self.span_emb_size, 'ReLU'))
            self.biaffine = Biaffine(self.span_emb_size, self.label_space_size - 1)
        self.loss_reduction = config.loss_reduction
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.span_width_embedding.weight)
        # init.xavier_uniform_(self.context_projective_layer.weight)
        # initializer_1d(self.context_projective_layer.bias, init.xavier_uniform_)

        for layer in self.arg_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.arg_unary_score_projection.weight)
        initializer_1d(self.arg_unary_score_projection.bias, init.xavier_uniform_)

        for layer in self.pred_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.pred_unary_score_projection.weight)
        initializer_1d(self.pred_unary_score_projection.bias, init.xavier_uniform_)

        for layer in self.srl_unary_score_layers:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)
        init.xavier_uniform_(self.srl_unary_score_projection.weight)
        initializer_1d(self.srl_unary_score_projection.bias, init.xavier_uniform_)
        return None

    def forward(self, hidden_states, batch, mask=None):
        gold_arg_ends, gold_arg_labels, gold_arg_starts, gold_predicates, masks, sent_lengths = SpanRankingSRLModel.unpack(
            batch, mask=mask, training=self.training)
        return self.decode(hidden_states, sent_lengths, masks, gold_arg_starts, gold_arg_ends, gold_arg_labels,
                           gold_predicates)

    @staticmethod
    def get_candidate_spans(sent_lengths: torch.Tensor, max_sent_length, max_arg_width):
        num_sentences = len(sent_lengths)
        device = sent_lengths.device
        candidate_starts = torch.arange(0, max_sent_length, device=device).expand(num_sentences, max_arg_width, -1)
        candidate_width = torch.arange(0, max_arg_width, device=device).view(1, -1, 1)
        candidate_ends = candidate_starts + candidate_width

        candidate_starts = candidate_starts.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        candidate_ends = candidate_ends.contiguous().view(num_sentences, max_sent_length * max_arg_width)
        actual_sent_lengths = sent_lengths.view(-1, 1).expand(-1, max_sent_length * max_arg_width)
        candidate_mask = candidate_ends < actual_sent_lengths

        candidate_starts = candidate_starts * candidate_mask
        candidate_ends = candidate_ends * candidate_mask
        return candidate_starts, candidate_ends, candidate_mask

    @staticmethod
    def exclusive_cumsum(input: torch.Tensor, exclusive=True):
        """

        Args:
          input: input is the sentence lengths tensor.
          exclusive: exclude the last sentence length (Default value = True)
          input(torch.Tensor :): 
          input: torch.Tensor: 

        Returns:

        
        """
        assert exclusive is True
        if exclusive is True:
            exclusive_sent_lengths = input.new_zeros(1, dtype=torch.long)
            result = torch.cumsum(torch.cat([exclusive_sent_lengths, input], 0)[:-1], 0).view(-1, 1)
        else:
            result = torch.cumsum(input, 0).view(-1, 1)
        return result

    def flatten_emb(self, emb):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        assert len(emb.size()) == 3
        flatted_emb = emb.contiguous().view(num_sentences * max_sentence_length, -1)
        return flatted_emb

    def flatten_emb_in_sentence(self, emb, batch_sentences_mask):
        num_sentences, max_sentence_length = emb.size()[0], emb.size()[1]
        flatted_emb = self.flatten_emb(emb)
        return flatted_emb[batch_sentences_mask.reshape(num_sentences * max_sentence_length)]

    def get_span_emb(self, flatted_context_emb, flatted_candidate_starts, flatted_candidate_ends,
                     config, dropout=0.0):
        batch_word_num = flatted_context_emb.size()[0]
        # gather slices from embeddings according to indices
        span_start_emb = flatted_context_emb[flatted_candidate_starts]
        span_end_emb = flatted_context_emb[flatted_candidate_ends]
        span_emb_feature_list = [span_start_emb, span_end_emb]  # store the span vector representations for span rep.

        span_width = 1 + flatted_candidate_ends - flatted_candidate_starts  # [num_spans], generate the span width
        max_arg_width = config.max_arg_width

        # get the span width feature emb
        span_width_index = span_width - 1
        span_width_emb = self.span_width_embedding(span_width_index)
        span_width_emb = F.dropout(span_width_emb, dropout, self.training)
        span_emb_feature_list.append(span_width_emb)

        """head features"""
        cpu_flatted_candidte_starts = flatted_candidate_starts
        span_indices = torch.arange(0, max_arg_width, device=flatted_context_emb.device).view(1, -1) + \
                       cpu_flatted_candidte_starts.view(-1, 1)  # For all the i, where i in [begin, ..i, end] for span
        # reset the position index to the batch_word_num index with index - 1
        span_indices = torch.clamp(span_indices, max=batch_word_num - 1)
        num_spans, spans_width = span_indices.size()[0], span_indices.size()[1]
        flatted_span_indices = span_indices.view(-1)  # so Huge!!!, column is the span?
        # if torch.cuda.is_available():
        flatted_span_indices = flatted_span_indices
        span_text_emb = flatted_context_emb.index_select(0, flatted_span_indices).view(num_spans, spans_width, -1)
        span_indices_mask = util.lengths_to_mask(span_width, max_len=max_arg_width)
        # project context output to num head
        # head_scores = self.context_projective_layer.forward(flatted_context_emb)
        # get span attention
        # span_attention = head_scores.index_select(0, flatted_span_indices).view(num_spans, spans_width)
        # span_attention = torch.add(span_attention, expanded_span_indices_log_mask).unsqueeze(2)  # control the span len
        # span_attention = F.softmax(span_attention, dim=1)
        span_text_emb = span_text_emb * span_indices_mask.unsqueeze(2).expand(-1, -1, span_text_emb.size()[-1])
        span_head_emb = torch.mean(span_text_emb, 1)
        span_emb_feature_list.append(span_head_emb)

        span_emb = torch.cat(span_emb_feature_list, 1)
        return span_emb, None, span_text_emb, span_indices, span_indices_mask

    def get_arg_unary_scores(self, span_emb):
        """Compute span score with FFNN(span embedding)

        Args:
          span_emb: tensor of [num_sentences, num_spans, emb_size]
          config: param dropout:
          num_labels: param name:

        Returns:

        
        """
        input = span_emb
        for i, ffnn in enumerate(self.arg_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.arg_dropout_layers[i].forward(input)
        output = self.arg_unary_score_projection.forward(input)
        return output

    def get_pred_unary_scores(self, span_emb):
        input = span_emb
        for i, ffnn in enumerate(self.pred_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.pred_dropout_layers[i].forward(input)
        output = self.pred_unary_score_projection.forward(input)
        return output

    def extract_spans(self, candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length,
                      sort_spans, enforce_non_crossing):
        """extract the topk span indices

        Args:
          candidate_scores: param candidate_starts:
          candidate_ends: param topk: [num_sentences]
          max_sentence_length: param sort_spans:
          enforce_non_crossing: return: indices [num_sentences, max_num_predictions]
          candidate_starts: 
          topk: 
          sort_spans: 

        Returns:

        
        """
        # num_sentences = candidate_scores.size()[0]
        # num_input_spans = candidate_scores.size()[1]
        max_num_output_spans = int(torch.max(topk))
        indices = [score.topk(k)[1] for score, k in zip(candidate_scores, topk)]
        output_span_indices_tensor = [F.pad(item, [0, max_num_output_spans - item.size()[0]], value=item[-1])
                                      for item in indices]
        output_span_indices_tensor = torch.stack(output_span_indices_tensor)
        return output_span_indices_tensor

    def batch_index_select(self, emb, indices):
        num_sentences = emb.size()[0]
        max_sent_length = emb.size()[1]
        flatten_emb = self.flatten_emb(emb)
        offset = (torch.arange(0, num_sentences, device=emb.device) * max_sent_length).unsqueeze(1)
        return torch.index_select(flatten_emb, 0, (indices + offset).view(-1)) \
            .view(indices.size()[0], indices.size()[1], -1)

    def get_batch_topk(self, candidate_starts: torch.Tensor, candidate_ends, candidate_scores, topk_ratio, text_len,
                       max_sentence_length, sort_spans=False, enforce_non_crossing=True):
        num_sentences = candidate_starts.size()[0]
        max_sentence_length = candidate_starts.size()[1]

        topk = torch.floor(text_len.to(torch.float) * topk_ratio).to(torch.long)
        topk = torch.max(topk, torch.ones(num_sentences, device=candidate_starts.device, dtype=torch.long))

        # this part should be implemented with C++
        predicted_indices = self.extract_spans(candidate_scores, candidate_starts, candidate_ends, topk,
                                               max_sentence_length, sort_spans, enforce_non_crossing)
        predicted_starts = torch.gather(candidate_starts, 1, predicted_indices)
        predicted_ends = torch.gather(candidate_ends, 1, predicted_indices)
        predicted_scores = torch.gather(candidate_scores, 1, predicted_indices)
        return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices

    def get_dense_span_labels(self, span_starts, span_ends, span_labels, max_sentence_length,
                              span_parents=None):
        num_sentences = span_starts.size()[0]
        max_spans_num = span_starts.size()[1]

        # span_starts = span_starts + 1 - (span_labels > 0).to(torch.long)
        span_starts[(span_labels == 0) & (span_starts < max_sentence_length - 1)] += 1  # make start > end
        sentence_indices = torch.arange(0, num_sentences, device=span_starts.device).unsqueeze(1).expand(-1,
                                                                                                         max_spans_num)

        sparse_indices = torch.cat([sentence_indices.unsqueeze(2), span_starts.unsqueeze(2), span_ends.unsqueeze(2)],
                                   dim=2)
        if span_parents is not None:  # semantic span predicate offset
            sparse_indices = torch.cat([sparse_indices, span_parents.unsqueeze(2)], 2)

        rank = 3 if span_parents is None else 4
        dense_labels = torch.sparse.LongTensor(sparse_indices.view(num_sentences * max_spans_num, rank).t(),
                                               span_labels.view(-1),
                                               torch.Size([num_sentences] + [max_sentence_length] * (rank - 1))) \
            .to_dense()
        return dense_labels

    @staticmethod
    def gather_4d(params, indices):
        assert len(params.size()) == 4 and len(indices) == 4
        indices_a, indices_b, indices_c, indices_d = indices
        result = params[indices_a, indices_b, indices_c, indices_d]
        return result

    def get_srl_labels(self,
                       arg_starts,
                       arg_ends,
                       predicates,
                       gold_predicates,
                       gold_arg_starts,
                       gold_arg_ends,
                       gold_arg_labels,
                       max_sentence_length
                       ):
        num_sentences = arg_starts.size()[0]
        max_arg_num = arg_starts.size()[1]
        max_pred_num = predicates.size()[1]

        sentence_indices_2d = torch.arange(0, num_sentences, device=arg_starts.device).unsqueeze(1).unsqueeze(2).expand(
            -1, max_arg_num, max_pred_num)
        expanded_arg_starts = arg_starts.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_arg_ends = arg_ends.unsqueeze(2).expand(-1, -1, max_pred_num)
        expanded_predicates = predicates.unsqueeze(1).expand(-1, max_arg_num, -1)

        dense_srl_labels = self.get_dense_span_labels(gold_arg_starts,
                                                      gold_arg_ends,
                                                      gold_arg_labels,
                                                      max_sentence_length, span_parents=gold_predicates)  # ans
        srl_labels = self.gather_4d(dense_srl_labels,
                                    [sentence_indices_2d, expanded_arg_starts, expanded_arg_ends, expanded_predicates])
        return srl_labels

    def get_srl_unary_scores(self, span_emb):
        input = span_emb
        for i, ffnn in enumerate(self.srl_unary_score_layers):
            input = F.relu(ffnn.forward(input))
            input = self.srl_dropout_layers[i].forward(input)
        output = self.srl_unary_score_projection.forward(input)
        return output

    def get_srl_scores(self, arg_emb, pred_emb, arg_scores, pred_scores, num_labels, config, dropout):
        num_sentences = arg_emb.size()[0]
        num_args = arg_emb.size()[1]  # [batch_size, max_arg_num, arg_emb_size]
        num_preds = pred_emb.size()[1]  # [batch_size, max_pred_num, pred_emb_size]

        unsqueezed_arg_emb = arg_emb.unsqueeze(2)
        unsqueezed_pred_emb = pred_emb.unsqueeze(1)
        expanded_arg_emb = unsqueezed_arg_emb.expand(-1, -1, num_preds, -1)
        expanded_pred_emb = unsqueezed_pred_emb.expand(-1, num_args, -1, -1)
        pair_emb_list = [expanded_arg_emb, expanded_pred_emb]
        pair_emb = torch.cat(pair_emb_list, 3)  # concatenate the argument emb and pre emb
        pair_emb_size = pair_emb.size()[3]
        flat_pair_emb = pair_emb.view(num_sentences * num_args * num_preds, pair_emb_size)
        # get unary scores
        flat_srl_scores = self.get_srl_unary_scores(flat_pair_emb)
        srl_scores = flat_srl_scores.view(num_sentences, num_args, num_preds, -1)
        if self.config.use_biaffine:
            srl_scores += self.biaffine(arg_emb, self.predicate_scale(pred_emb)).permute([0, 2, 3, 1])
        unsqueezed_arg_scores, unsqueezed_pred_scores = \
            arg_scores.unsqueeze(2).unsqueeze(3), pred_scores.unsqueeze(1).unsqueeze(3)
        srl_scores = srl_scores + unsqueezed_arg_scores + unsqueezed_pred_scores
        dummy_scores = torch.zeros([num_sentences, num_args, num_preds, 1], device=arg_emb.device)
        srl_scores = torch.cat([dummy_scores, srl_scores], 3)
        return srl_scores

    def get_srl_softmax_loss(self, srl_scores, srl_labels, num_predicted_args, num_predicted_preds):
        srl_loss_mask = self.get_srl_loss_mask(srl_scores, num_predicted_args, num_predicted_preds)

        loss = torch.nn.functional.cross_entropy(srl_scores[srl_loss_mask], srl_labels[srl_loss_mask],
                                                 reduction=self.loss_reduction)
        return loss, srl_loss_mask

    def get_srl_loss_mask(self, srl_scores, num_predicted_args, num_predicted_preds):
        max_num_arg = srl_scores.size()[1]
        max_num_pred = srl_scores.size()[2]
        # num_predicted_args, 1D tensor; max_num_arg: a int variable means the gold ans's max arg number
        args_mask = util.lengths_to_mask(num_predicted_args, max_num_arg)
        pred_mask = util.lengths_to_mask(num_predicted_preds, max_num_pred)
        srl_loss_mask = args_mask.unsqueeze(2) & pred_mask.unsqueeze(1)
        return srl_loss_mask

    def decode(self, contextualized_embeddings, sent_lengths, masks, gold_arg_starts, gold_arg_ends, gold_arg_labels,
               gold_predicates):
        num_sentences, max_sent_length = masks.size()
        device = sent_lengths.device
        """generate candidate spans with argument pruning"""
        # candidate_starts [num_sentences, max_sent_length * max_arg_width]
        candidate_starts, candidate_ends, candidate_mask = self.get_candidate_spans(
            sent_lengths, max_sent_length, self.config.max_arg_width)
        flatted_candidate_mask = candidate_mask.view(-1)
        batch_word_offset = self.exclusive_cumsum(sent_lengths)  # get the word offset in a batch
        # choose the flatted_candidate_starts with the actual existing positions, i.e. exclude the illegal starts
        flatted_candidate_starts = candidate_starts + batch_word_offset
        flatted_candidate_starts = flatted_candidate_starts.view(-1)[flatted_candidate_mask].to(torch.long)
        flatted_candidate_ends = candidate_ends + batch_word_offset
        flatted_candidate_ends = flatted_candidate_ends.view(-1)[flatted_candidate_mask].to(torch.long)
        # flatten the lstm output according to the sentence mask, i.e. exclude the illegal (padding) lstm output
        flatted_context_output = self.flatten_emb_in_sentence(contextualized_embeddings, masks)
        """generate the span embedding"""
        candidate_span_emb, head_scores, span_head_emb, head_indices, head_indices_log_mask = self.get_span_emb(
            flatted_context_output, flatted_candidate_starts, flatted_candidate_ends,
            self.config, dropout=self.dropout)
        """Get the span ids"""
        candidate_span_number = candidate_span_emb.size()[0]
        max_candidate_spans_num_per_sentence = candidate_mask.size()[1]
        sparse_indices = candidate_mask.nonzero(as_tuple=False)
        sparse_values = torch.arange(0, candidate_span_number, device=device)
        candidate_span_ids = torch.sparse.FloatTensor(sparse_indices.t(), sparse_values,
                                                      torch.Size([num_sentences,
                                                                  max_candidate_spans_num_per_sentence])).to_dense()
        spans_log_mask = torch.log(candidate_mask.to(torch.float))
        predict_dict = {"candidate_starts": candidate_starts, "candidate_ends": candidate_ends,
                        "head_scores": head_scores}
        """Get unary scores and topk of candidate argument spans."""
        flatted_candidate_arg_scores = self.get_arg_unary_scores(candidate_span_emb)
        candidate_arg_scores = flatted_candidate_arg_scores.index_select(0, candidate_span_ids.view(-1)) \
            .view(candidate_span_ids.size()[0], candidate_span_ids.size()[1])
        candidate_arg_scores = candidate_arg_scores + spans_log_mask
        arg_starts, arg_ends, arg_scores, num_args, top_arg_indices = \
            self.get_batch_topk(candidate_starts, candidate_ends, candidate_arg_scores,
                                self.config.argument_ratio, sent_lengths, max_sent_length,
                                sort_spans=False, enforce_non_crossing=False)
        """Get the candidate predicate"""
        candidate_pred_ids = torch.arange(0, max_sent_length, device=device).unsqueeze(0).expand(num_sentences, -1)
        candidate_pred_emb = contextualized_embeddings
        candidate_pred_scores = self.get_pred_unary_scores(candidate_pred_emb)
        candidate_pred_scores = candidate_pred_scores + torch.log(masks.to(torch.float).unsqueeze(2))
        candidate_pred_scores = candidate_pred_scores.squeeze(2)
        if self.use_gold_predicates is True:
            predicates = gold_predicates[0]
            num_preds = gold_predicates[1]
            pred_scores = torch.zeros_like(predicates)
            top_pred_indices = predicates
        else:
            predicates, _, pred_scores, num_preds, top_pred_indices = self.get_batch_topk(
                candidate_pred_ids, candidate_pred_ids, candidate_pred_scores, self.config.predicate_ratio,
                sent_lengths, max_sent_length,
                sort_spans=False, enforce_non_crossing=False)
        """Get top arg embeddings"""
        arg_span_indices = torch.gather(candidate_span_ids, 1, top_arg_indices)  # [num_sentences, max_num_args]
        arg_emb = candidate_span_emb.index_select(0, arg_span_indices.view(-1)).view(
            arg_span_indices.size()[0], arg_span_indices.size()[1], -1
        )  # [num_sentences, max_num_args, emb]
        """Get top predicate embeddings"""
        pred_emb = self.batch_index_select(candidate_pred_emb,
                                           top_pred_indices)  # [num_sentences, max_num_preds, emb]
        """Get the srl scores according to the arg emb and pre emb."""
        srl_scores = self.get_srl_scores(arg_emb, pred_emb, arg_scores, pred_scores, self.label_space_size, self.config,
                                         self.dropout)  # [num_sentences, max_num_args, max_num_preds, num_labels]
        if gold_arg_labels is not None:
            """Get the answers according to the labels"""
            srl_labels = self.get_srl_labels(arg_starts, arg_ends, predicates, gold_predicates, gold_arg_starts,
                                             gold_arg_ends, gold_arg_labels, max_sent_length)

            """Compute the srl loss"""
            srl_loss, srl_mask = self.get_srl_softmax_loss(srl_scores, srl_labels, num_args, num_preds)
            predict_dict.update({
                'srl_mask': srl_mask,
                'loss': srl_loss
            })
        else:
            predict_dict['srl_mask'] = self.get_srl_loss_mask(srl_scores, num_args, num_preds)
        predict_dict.update({
            "candidate_arg_scores": candidate_arg_scores,
            "candidate_pred_scores": candidate_pred_scores,
            "predicates": predicates,
            "arg_starts": arg_starts,
            "arg_ends": arg_ends,
            "arg_scores": arg_scores,
            "pred_scores": pred_scores,
            "num_args": num_args,
            "num_preds": num_preds,
            "arg_labels": torch.max(srl_scores, 1)[1],  # [num_sentences, num_args, num_preds]
            "srl_scores": srl_scores,
        })
        return predict_dict


class SpanRankingSRLModel(nn.Module):

    def __init__(self, config, embed: torch.nn.Module, context_layer: torch.nn.Module, label_space_size):
        super(SpanRankingSRLModel, self).__init__()
        self.config = config
        self.dropout = float(config.dropout)
        self.lexical_dropout = float(self.config.lexical_dropout)
        self.label_space_size = label_space_size

        # Initialize layers and parameters
        self.word_embedding_dim = embed.get_output_dim()  # get the embedding dim
        self.embed = embed
        # Initialize context layer
        self.context_layer = context_layer
        context_layer_output_dim = context_layer.get_output_dim()
        self.decoder = SpanRankingSRLDecoder(context_layer_output_dim, label_space_size, config)

    def forward(self,
                batch: Dict[str, torch.Tensor]
                ):
        gold_arg_ends, gold_arg_labels, gold_arg_starts, gold_predicates, masks, sent_lengths = \
            self.unpack(batch, training=self.training)

        context_embeddings = self.embed(batch)
        context_embeddings = F.dropout(context_embeddings, self.lexical_dropout, self.training)
        contextualized_embeddings = self.context_layer(context_embeddings, masks)

        return self.decoder.decode(contextualized_embeddings, sent_lengths, masks, gold_arg_starts, gold_arg_ends,
                                   gold_arg_labels, gold_predicates)

    @staticmethod
    def unpack(batch, mask=None, training=False):
        keys = 'token_length', 'predicate_offset', 'argument_begin_offset', 'argument_end_offset', 'srl_label_id'
        sent_lengths, gold_predicates, gold_arg_starts, gold_arg_ends, gold_arg_labels = [batch.get(k, None) for k in
                                                                                          keys]
        if mask is None:
            mask = util.lengths_to_mask(sent_lengths)
        # elif not training:
        #     sent_lengths = mask.sum(dim=1)
        return gold_arg_ends, gold_arg_labels, gold_arg_starts, gold_predicates, mask, sent_lengths
