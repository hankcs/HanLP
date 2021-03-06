# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-15 21:22
from collections import defaultdict
from typing import Tuple, Union

import torch
from torch.nn import functional as F

from hanlp.components.parsers.ud import udify_util as util
from hanlp.layers.transformers.pt_imports import PreTrainedModel, optimization, AdamW, \
    get_linear_schedule_with_warmup


def transformer_encode(transformer: PreTrainedModel,
                       input_ids,
                       attention_mask=None,
                       token_type_ids=None,
                       token_span=None,
                       layer_range: Union[int, Tuple[int, int]] = 0,
                       max_sequence_length=None,
                       average_subwords=False,
                       ret_raw_hidden_states=False):
    """Run transformer and pool its outputs.

    Args:
        transformer: A transformer model.
        input_ids: Indices of subwords.
        attention_mask: Mask for these subwords.
        token_type_ids: Type ids for each subword.
        token_span: The spans of tokens.
        layer_range: The range of layers to use. Note that the 0-th layer means embedding layer, so the last 3 layers
                    of a 12-layer BERT will be (10, 13).
        max_sequence_length: The maximum sequence length. Sequence longer than this will be handled by sliding
                    window.
         average_subwords: ``True`` to average subword representations.
        ret_raw_hidden_states: ``True`` to return hidden states of each layer.

    Returns:
        Pooled outputs.

    """
    if max_sequence_length and input_ids.size(-1) > max_sequence_length:
        # TODO: split token type ids in transformer_sliding_window if token type ids are not always 1
        outputs = transformer_sliding_window(transformer, input_ids, max_pieces=max_sequence_length)
    else:
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        if transformer.config.output_hidden_states:
            outputs = transformer(input_ids, attention_mask, token_type_ids)[-1]
        else:
            outputs = transformer(input_ids, attention_mask, token_type_ids)[0]
    if transformer.config.output_hidden_states:
        if isinstance(layer_range, int):
            outputs = outputs[layer_range:]
        else:
            outputs = outputs[layer_range[0], layer_range[1]]
        # Slow pick
        # hs = []
        # for h in outputs:
        #     hs.append(pick_tensor_for_each_token(h, token_span, average_subwords))
        # Fast pick
        if not isinstance(outputs, torch.Tensor):
            x = torch.stack(outputs)
        else:
            x = outputs
        L, B, T, F = x.size()
        x = x.flatten(end_dim=1)
        # tile token_span as x
        if token_span is not None:
            token_span = token_span.repeat(L, 1, 1)
        hs = pick_tensor_for_each_token(x, token_span, average_subwords).view(L, B, -1, F)
        if ret_raw_hidden_states:
            return hs, outputs
        return hs
    else:
        if ret_raw_hidden_states:
            return pick_tensor_for_each_token(outputs, token_span, average_subwords), outputs
        return pick_tensor_for_each_token(outputs, token_span, average_subwords)


def pick_tensor_for_each_token(h, token_span, average_subwords):
    if token_span is None:
        return h
    if average_subwords and token_span.size(-1) > 1:
        batch_size = h.size(0)
        h_span = h.gather(1, token_span.view(batch_size, -1).unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        h_span = h_span.view(batch_size, *token_span.shape[1:], -1)
        n_sub_tokens = token_span.ne(0)
        n_sub_tokens[:, 0, 0] = True
        h_span = (h_span * n_sub_tokens.unsqueeze(-1)).sum(2)
        n_sub_tokens = n_sub_tokens.sum(-1).unsqueeze(-1)
        zero_mask = n_sub_tokens == 0
        if torch.any(zero_mask):
            n_sub_tokens[zero_mask] = 1  # avoid dividing by zero
        embed = h_span / n_sub_tokens
    else:
        embed = h.gather(1, token_span[:, :, 0].unsqueeze(-1).expand(-1, -1, h.size(-1)))
    return embed


def transformer_sliding_window(transformer: PreTrainedModel,
                               input_ids: torch.LongTensor,
                               input_mask=None,
                               offsets: torch.LongTensor = None,
                               token_type_ids: torch.LongTensor = None,
                               max_pieces=512,
                               start_tokens: int = 1,
                               end_tokens: int = 1,
                               ret_cls=None,
                               ) -> torch.Tensor:
    """

    Args:
      transformer:
      input_ids: torch.LongTensor: 
      input_mask:  (Default value = None)
      offsets: torch.LongTensor:  (Default value = None)
      token_type_ids: torch.LongTensor:  (Default value = None)
      max_pieces:  (Default value = 512)
      start_tokens: int:  (Default value = 1)
      end_tokens: int:  (Default value = 1)
      ret_cls:  (Default value = None)

    Returns:

    
    """
    # pylint: disable=arguments-differ
    batch_size, full_seq_len = input_ids.size(0), input_ids.size(-1)
    initial_dims = list(input_ids.shape[:-1])

    # The embedder may receive an input tensor that has a sequence length longer than can
    # be fit. In that case, we should expect the wordpiece indexer to create padded windows
    # of length `max_pieces` for us, and have them concatenated into one long sequence.
    # E.g., "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ..."
    # We can then split the sequence into sub-sequences of that length, and concatenate them
    # along the batch dimension so we effectively have one huge batch of partial sentences.
    # This can then be fed into BERT without any sentence length issues. Keep in mind
    # that the memory consumption can dramatically increase for large batches with extremely
    # long sentences.
    needs_split = full_seq_len > max_pieces
    if needs_split:
        input_ids = split_to_sliding_window(input_ids, max_pieces)

    # if token_type_ids is None:
    #     token_type_ids = torch.zeros_like(input_ids)
    if input_mask is None:
        input_mask = (input_ids != 0).long()

    # input_ids may have extra dimensions, so we reshape down to 2-d
    # before calling the BERT model and then reshape back at the end.
    outputs = transformer(input_ids=util.combine_initial_dims_to_1d_or_2d(input_ids),
                          # token_type_ids=util.combine_initial_dims_to_1d_or_2d(token_type_ids),
                          attention_mask=util.combine_initial_dims_to_1d_or_2d(input_mask)).to_tuple()
    if len(outputs) == 3:
        all_encoder_layers = outputs.hidden_states
        all_encoder_layers = torch.stack(all_encoder_layers)
    elif len(outputs) == 2:
        all_encoder_layers, _ = outputs[:2]
    else:
        all_encoder_layers = outputs[0]

    if needs_split:
        if ret_cls is not None:
            cls_mask = input_ids[:, 0] == input_ids[0][0]
            cls_hidden = all_encoder_layers[:, 0, :]
            if ret_cls == 'max':
                cls_hidden[~cls_mask] = -1e20
            else:
                cls_hidden[~cls_mask] = 0
            cls_mask = cls_mask.view(-1, batch_size).transpose(0, 1)
            cls_hidden = cls_hidden.reshape(cls_mask.size(1), batch_size, -1).transpose(0, 1)
            if ret_cls == 'max':
                cls_hidden = cls_hidden.max(1)[0]
            elif ret_cls == 'raw':
                return cls_hidden, cls_mask
            else:
                cls_hidden = torch.sum(cls_hidden, dim=1)
                cls_hidden /= torch.sum(cls_mask, dim=1, keepdim=True)
            return cls_hidden
        else:
            recombined_embeddings, select_indices = restore_from_sliding_window(all_encoder_layers, batch_size,
                                                                                max_pieces, full_seq_len, start_tokens,
                                                                                end_tokens)

            initial_dims.append(len(select_indices))
    else:
        recombined_embeddings = all_encoder_layers

    # Recombine the outputs of all layers
    # (layers, batch_size * d1 * ... * dn, sequence_length, embedding_dim)
    # recombined = torch.cat(combined, dim=2)
    # input_mask = (recombined_embeddings != 0).long()

    # At this point, mix is (batch_size * d1 * ... * dn, sequence_length, embedding_dim)

    if offsets is None:
        # Resize to (batch_size, d1, ..., dn, sequence_length, embedding_dim)
        dims = initial_dims if needs_split else input_ids.size()
        layers = util.uncombine_initial_dims(recombined_embeddings, dims)
    else:
        # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
        offsets2d = util.combine_initial_dims_to_1d_or_2d(offsets)
        # now offsets is (batch_size * d1 * ... * dn, orig_sequence_length)
        range_vector = util.get_range_vector(offsets2d.size(0),
                                             device=util.get_device_of(recombined_embeddings)).unsqueeze(1)
        # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
        selected_embeddings = recombined_embeddings[:, range_vector, offsets2d]

        layers = util.uncombine_initial_dims(selected_embeddings, offsets.size())

    return layers


def split_to_sliding_window(input_ids, max_pieces):
    # Split the flattened list by the window size, `max_pieces`
    split_input_ids = list(input_ids.split(max_pieces, dim=-1))
    # We want all sequences to be the same length, so pad the last sequence
    last_window_size = split_input_ids[-1].size(-1)
    padding_amount = max_pieces - last_window_size
    split_input_ids[-1] = F.pad(split_input_ids[-1], pad=[0, padding_amount], value=0)
    # Now combine the sequences along the batch dimension
    input_ids = torch.cat(split_input_ids, dim=0)
    return input_ids


def restore_from_sliding_window(all_encoder_layers, batch_size, max_pieces, full_seq_len, start_tokens, end_tokens):
    # First, unpack the output embeddings into one long sequence again
    unpacked_embeddings = torch.split(all_encoder_layers, batch_size, dim=-3)
    unpacked_embeddings = torch.cat(unpacked_embeddings, dim=-2)
    # Next, select indices of the sequence such that it will result in embeddings representing the original
    # sentence. To capture maximal context, the indices will be the middle part of each embedded window
    # sub-sequence (plus any leftover start and final edge windows), e.g.,
    #  0     1 2    3  4   5    6    7     8     9   10   11   12    13 14  15
    # "[CLS] I went to the very fine [SEP] [CLS] the very fine store to eat [SEP]"
    # with max_pieces = 8 should produce max context indices [2, 3, 4, 10, 11, 12] with additional start
    # and final windows with indices [0, 1] and [14, 15] respectively.
    # Find the stride as half the max pieces, ignoring the special start and end tokens
    # Calculate an offset to extract the centermost embeddings of each window
    stride = (max_pieces - start_tokens - end_tokens) // 2
    stride_offset = stride // 2 + start_tokens
    first_window = list(range(stride_offset))
    max_context_windows = [i for i in range(full_seq_len)
                           if stride_offset - 1 < i % max_pieces < stride_offset + stride]
    final_window_start = max_context_windows[-1] + 1
    final_window = list(range(final_window_start, full_seq_len))
    select_indices = first_window + max_context_windows + final_window
    select_indices = torch.LongTensor(select_indices).to(unpacked_embeddings.device)
    recombined_embeddings = unpacked_embeddings.index_select(-2, select_indices)
    return recombined_embeddings, select_indices


def build_optimizer_for_pretrained(model: torch.nn.Module,
                                   pretrained: torch.nn.Module,
                                   lr=1e-5,
                                   weight_decay=0.01,
                                   eps=1e-8,
                                   transformer_lr=None,
                                   transformer_weight_decay=None,
                                   no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
                                   **kwargs):
    if transformer_lr is None:
        transformer_lr = lr
    if transformer_weight_decay is None:
        transformer_weight_decay = weight_decay
    params = defaultdict(lambda: defaultdict(list))
    pretrained = set(pretrained.parameters())
    if isinstance(no_decay, tuple):
        def no_decay_fn(name):
            return any(nd in name for nd in no_decay)
    else:
        assert callable(no_decay), 'no_decay has to be callable or a tuple of str'
        no_decay_fn = no_decay
    for n, p in model.named_parameters():
        is_pretrained = 'pretrained' if p in pretrained else 'non_pretrained'
        is_no_decay = 'no_decay' if no_decay_fn(n) else 'decay'
        params[is_pretrained][is_no_decay].append(p)

    grouped_parameters = [
        {'params': params['pretrained']['decay'], 'weight_decay': transformer_weight_decay, 'lr': transformer_lr},
        {'params': params['pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': transformer_lr},
        {'params': params['non_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
        {'params': params['non_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': lr},
    ]

    return optimization.AdamW(
        grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        **kwargs)


def build_optimizer_scheduler_with_transformer(model: torch.nn.Module,
                                               transformer: torch.nn.Module,
                                               lr: float,
                                               transformer_lr: float,
                                               num_training_steps: int,
                                               warmup_steps: Union[float, int],
                                               weight_decay: float,
                                               adam_epsilon: float,
                                               no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight')):
    optimizer = build_optimizer_for_pretrained(model,
                                               transformer,
                                               lr,
                                               weight_decay,
                                               eps=adam_epsilon,
                                               transformer_lr=transformer_lr,
                                               no_decay=no_decay)
    if isinstance(warmup_steps, float):
        assert 0 < warmup_steps < 1, 'warmup_steps has to fall in range (0, 1) when it is float.'
        warmup_steps = num_training_steps * warmup_steps
    scheduler = optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    return optimizer, scheduler


def get_optimizers(
        model: torch.nn.Module,
        num_training_steps: int,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        warmup_steps=0.1,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Modified from https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L232
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well.
    """
    if isinstance(warmup_steps, float):
        assert 0 < warmup_steps < 1
        warmup_steps = int(num_training_steps * warmup_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def collect_decay_params(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
