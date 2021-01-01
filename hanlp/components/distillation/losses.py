# Adopted from https://github.com/airaria/TextBrewer
# Apache License Version 2.0

import torch
import torch.nn.functional as F

from hanlp_common.configurable import AutoConfigurable


def kd_mse_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the mse loss between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    loss = F.mse_loss(beta_logits_S, beta_logits_T)
    return loss


def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def att_mse_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the mse loss between `attention_S` and `attention_T`.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor mask: tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
        attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
        loss = F.mse_loss(attention_S_select, attention_T_select)
    else:
        mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
        valid_count = torch.pow(mask.sum(dim=2), 2).sum()
        loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
            2)).sum() / valid_count
    return loss


def att_mse_sum_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the mse loss between `attention_S` and `attention_T`. 
    * If the the shape is (*batch_size*, *num_heads*, *length*, *length*), sums along the `num_heads` dimension and then calcuates the mse loss between the two matrices.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.Tensor mask:     tensor of shape  (*batch_size*, *length*)
    '''
    if len(attention_S.size()) == 4:
        attention_T = attention_T.sum(dim=1)
        attention_S = attention_S.sum(dim=1)
    if mask is None:
        attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
        attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
        loss = F.mse_loss(attention_S_select, attention_T_select)
    else:
        mask = mask.to(attention_S)
        valid_count = torch.pow(mask.sum(dim=1), 2).sum()
        loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
            1)).sum() / valid_count
    return loss


def att_ce_loss(attention_S, attention_T, mask=None):
    '''

    * Calculates the cross-entropy loss between `attention_S` and `attention_T`, where softmax is to applied on ``dim=-1``.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    
    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor mask:     tensor of shape  (*batch_size*, *length*)
    '''
    probs_T = F.softmax(attention_T, dim=-1)
    if mask is None:
        probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
        loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
    else:
        mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
        loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(2)).sum(
            dim=-1) * mask).sum() / mask.sum()
    return loss


def att_ce_mean_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the cross-entropy loss between `attention_S` and `attention_T`, where softmax is to applied on ``dim=-1``.
    * If the shape is (*batch_size*, *num_heads*, *length*, *length*), averages over dimension `num_heads` and then computes cross-entropy loss between the two matrics.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    
    :param torch.tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.tensor mask:     tensor of shape  (*batch_size*, *length*)
    '''
    if len(attention_S.size()) == 4:
        attention_S = attention_S.mean(dim=1)  # (bs, len, len)
        attention_T = attention_T.mean(dim=1)
    probs_T = F.softmax(attention_T, dim=-1)
    if mask is None:
        probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
        loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
    else:
        mask = mask.to(attention_S)
        loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(1)).sum(
            dim=-1) * mask).sum() / mask.sum()
    return loss


def hid_mse_loss(state_S, state_T, mask=None):
    '''
    * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        loss = F.mse_loss(state_S, state_T)
    else:
        mask = mask.to(state_S)
        valid_count = mask.sum() * state_S.size(-1)
        loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
    return loss


def cos_loss(state_S, state_T, mask=None):
    '''
    * Computes the cosine similarity loss between the inputs. This is the loss used in DistilBERT, see `DistilBERT <https://arxiv.org/abs/1910.01108>`_
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        state_S = state_S.view(-1, state_S.size(-1))
        state_T = state_T.view(-1, state_T.size(-1))
    else:
        mask = mask.to(state_S).unsqueeze(-1).expand_as(state_S)  # (bs,len,dim)
        state_S = torch.masked_select(state_S, mask).view(-1, mask.size(-1))  # (bs * select, dim)
        state_T = torch.masked_select(state_T, mask).view(-1, mask.size(-1))  # (bs * select, dim)

    target = state_S.new(state_S.size(0)).fill_(1)
    loss = F.cosine_embedding_loss(state_S, state_T, target, reduction='mean')
    return loss


def pkd_loss(state_S, state_T, mask=None):
    '''
    * Computes normalized vector mse loss at position 0 along `length` dimension. This is the loss used in BERT-PKD, see `Patient Knowledge Distillation for BERT Model Compression <https://arxiv.org/abs/1908.09355>`_.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param mask: not used.
    '''

    cls_T = state_T[:, 0]  # (batch_size, hidden_dim)
    cls_S = state_S[:, 0]  # (batch_size, hidden_dim)
    normed_cls_T = cls_T / torch.norm(cls_T, dim=1, keepdim=True)
    normed_cls_S = cls_S / torch.norm(cls_S, dim=1, keepdim=True)
    loss = (normed_cls_S - normed_cls_T).pow(2).sum(dim=-1).mean()
    return loss


def fsp_loss(state_S, state_T, mask=None):
    r'''
    * Takes in two lists of matrics `state_S` and `state_T`. Each list contains two matrices of the shape (*batch_size*, *length*, *hidden_size*). Computes the similarity matrix between the two matrices in `state_S` ( with the resulting shape (*batch_size*, *hidden_size*, *hidden_size*) ) and the ones in B ( with the resulting shape (*batch_size*, *hidden_size*, *hidden_size*) ), then computes the mse loss between the similarity matrices:

    .. math::

        loss = mean((S_{1}^T \cdot S_{2} - T_{1}^T \cdot T_{2})^2)

    * It is a Variant of FSP loss in `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning <http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf>`_.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.tensor state_S: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor state_T: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor mask:    tensor of the shape  (*batch_size*, *length*)

    Example in `intermediate_matches`::

        intermediate_matches = [
        {'layer_T':[0,0], 'layer_S':[0,0], 'feature':'hidden','loss': 'fsp', 'weight' : 1, 'proj':['linear',384,768]},
        ...]
    '''
    if mask is None:
        state_S_0 = state_S[0]  # (batch_size , length, hidden_dim)
        state_S_1 = state_S[1]  # (batch_size,  length, hidden_dim)
        state_T_0 = state_T[0]
        state_T_1 = state_T[1]
        gram_S = torch.bmm(state_S_0.transpose(1, 2), state_S_1) / state_S_1.size(
            1)  # (batch_size, hidden_dim, hidden_dim)
        gram_T = torch.bmm(state_T_0.transpose(1, 2), state_T_1) / state_T_1.size(1)
    else:
        mask = mask.to(state_S[0]).unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True)
        state_S_0 = state_S[0] * mask
        state_S_1 = state_S[1] * mask
        state_T_0 = state_T[0] * mask
        state_T_1 = state_T[1] * mask
        gram_S = torch.bmm(state_S_0.transpose(1, 2), state_S_1) / lengths
        gram_T = torch.bmm(state_T_0.transpose(1, 2), state_T_1) / lengths
    loss = F.mse_loss(gram_S, gram_T)
    return loss


def mmd_loss(state_S, state_T, mask=None):
    r'''
    * Takes in two lists of matrices `state_S` and `state_T`. Each list contains 2 matrices of the shape (*batch_size*, *length*, *hidden_size*). `hidden_size` of matrices in `State_S` doesn't need to be the same as that of `state_T`. Computes the similarity matrix between the two matrices in `state_S` ( with the resulting shape (*batch_size*, *length*, *length*) ) and the ones in B ( with the resulting shape (*batch_size*, *length*, *length*) ), then computes the mse loss between the similarity matrices:
    
    .. math::

            loss = mean((S_{1} \cdot S_{2}^T - T_{1} \cdot T_{2}^T)^2)

    * It is a Variant of the NST loss in `Like What You Like: Knowledge Distill via Neuron Selectivity Transfer <https://arxiv.org/abs/1707.01219>`_
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

    :param torch.tensor state_S: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor state_T: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor mask:    tensor of the shape  (*batch_size*, *length*)

    Example in `intermediate_matches`::

        intermediate_matches = [
        {'layer_T':[0,0], 'layer_S':[0,0], 'feature':'hidden','loss': 'nst', 'weight' : 1},
        ...]
    '''
    state_S_0 = state_S[0]  # (batch_size , length, hidden_dim_S)
    state_S_1 = state_S[1]  # (batch_size , length, hidden_dim_S)
    state_T_0 = state_T[0]  # (batch_size , length, hidden_dim_T)
    state_T_1 = state_T[1]  # (batch_size , length, hidden_dim_T)
    if mask is None:
        gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(2)  # (batch_size, length, length)
        gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(2)
        loss = F.mse_loss(gram_S, gram_T)
    else:
        mask = mask.to(state_S[0])
        valid_count = torch.pow(mask.sum(dim=1), 2).sum()
        gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(1)  # (batch_size, length, length)
        gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(1)
        loss = (F.mse_loss(gram_S, gram_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
            1)).sum() / valid_count
    return loss


class KnowledgeDistillationLoss(AutoConfigurable):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        import sys
        thismodule = sys.modules[__name__]
        self._loss = getattr(thismodule, name)

    def __call__(self, *args, **kwargs):
        return self._loss(*args, **kwargs)
