# Adopted from https://github.com/KiroSummer/A_Syntax-aware_MTL_Framework_for_Chinese_SRL

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from hanlp.components.srl.span_rank.util import block_orth_normal_initializer


def get_tensor_np(t):
    return t.data.cpu().numpy()


def orthonormal_initializer(output_size, input_size):
    """adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py

    Args:
      output_size: 
      input_size: 

    Returns:

    
    """
    print((output_size, input_size))
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print(('Orthogonal pretrainer loss: %.2e' % loss))
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class DropoutLayer3D(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        super(DropoutLayer3D, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.drop_mask = torch.FloatTensor(self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = Variable(torch.bernoulli(self.drop_mask), requires_grad=False)
        if torch.cuda.is_available():
            self.drop_mask = self.drop_mask.cuda()

    def reset_dropout_mask(self, batch_size, length):
        self.drop_mask = torch.FloatTensor(batch_size, length, self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = Variable(torch.bernoulli(self.drop_mask), requires_grad=False)
        if torch.cuda.is_available():
            self.drop_mask = self.drop_mask.cuda()

    def forward(self, x):
        if self.training:
            return torch.mul(x, self.drop_mask)
        else:  # eval
            return x * (1.0 - self.dropout_rate)


class DropoutLayer(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.drop_mask = torch.Tensor(self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = torch.bernoulli(self.drop_mask)

    def reset_dropout_mask(self, batch_size):
        self.drop_mask = torch.Tensor(batch_size, self.input_size).fill_(1 - self.dropout_rate)
        self.drop_mask = torch.bernoulli(self.drop_mask)

    def forward(self, x):
        if self.training:
            return torch.mul(x, self.drop_mask.to(x.device))
        else:  # eval
            return x * (1.0 - self.dropout_rate)


class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)  # this kind of implementation is too tedious
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        # torch.bmm: Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)
        # view: Returns a new tensor with the same data as the self tensor but of a different size.
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self.in1_features) \
               + ', in2_features=' + str(self.in2_features) \
               + ', out_features=' + str(self.out_features) + ')'


class HighwayLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HighwayLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=6 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=5 * hidden_size,
                                   bias=False)
        self.reset_parameters()  # reset all the param in the MyLSTMCell

    def reset_parameters(self):
        weight_ih = block_orth_normal_initializer([self.input_size, ], [self.hidden_size] * 6)
        self.linear_ih.weight.data.copy_(weight_ih)

        weight_hh = block_orth_normal_initializer([self.hidden_size, ], [self.hidden_size] * 5)
        self.linear_hh.weight.data.copy_(weight_hh)
        # nn.init.constant(self.linear_hh.weight, 1.0)
        # nn.init.constant(self.linear_ih.weight, 1.0)

        nn.init.constant(self.linear_ih.bias, 0.0)

    def forward(self, x, mask=None, hx=None, dropout=None):
        assert mask is not None and hx is not None
        _h, _c = hx
        _x = self.linear_ih(x)  # compute the x
        preact = self.linear_hh(_h) + _x[:, :self.hidden_size * 5]

        i, f, o, t, j = preact.chunk(chunks=5, dim=1)
        i, f, o, t, j = F.sigmoid(i), F.sigmoid(f + 1.0), F.sigmoid(o), F.sigmoid(t), F.tanh(j)
        k = _x[:, self.hidden_size * 5:]

        c = f * _c + i * j
        c = mask * c + (1.0 - mask) * _c

        h = t * o * F.tanh(c) + (1.0 - t) * k
        if dropout is not None:
            h = dropout(h)
        h = mask * h + (1.0 - mask) * _h
        return h, c


class VariationalLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VariationalLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size + self.hidden_size, out_features=3 * hidden_size)
        self.reset_parameters()  # reset all the param in the MyLSTMCell

    def reset_parameters(self):
        weight = block_orth_normal_initializer([self.input_size + self.hidden_size, ], [self.hidden_size] * 3)
        self.linear.weight.data.copy_(weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x, mask=None, hx=None, dropout=None):
        assert mask is not None and hx is not None
        _h, _c = hx
        _h = dropout(_h)
        _x = self.linear(torch.cat([x, _h], 1))  # compute the x
        i, j, o = _x.chunk(3, dim=1)
        i = torch.sigmoid(i)
        c = (1.0 - i) * _c + i * torch.tanh(j)
        c = mask * c  # + (1.0 - mask) * _c
        h = torch.tanh(c) * torch.sigmoid(o)
        h = mask * h  # + (1.0 - mask) * _h

        return h, c


class VariationalLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(VariationalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells = []
        self.bcells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            if self.bidirectional:
                self.bcells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))

        self._all_weights = []
        for layer in range(num_layers):
            layer_params = (self.fcells[layer].weight_ih, self.fcells[layer].weight_hh, \
                            self.fcells[layer].bias_ih, self.fcells[layer].bias_hh)
            suffix = ''
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

            if self.bidirectional:
                layer_params = (self.bcells[layer].weight_ih, self.bcells[layer].weight_hh, \
                                self.bcells[layer].bias_ih, self.bcells[layer].bias_hh)
                suffix = '_reverse'
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):  # modified by kiro
        for name, param in self.named_parameters():
            print(name)
            if "weight" in name:
                # for i in range(4):
                # nn.init.orthogonal(self.__getattr__(name)[self.hidden_size*i:self.hidden_size*(i+1),:])
                nn.init.orthogonal(self.__getattr__(name))
            if "bias" in name:
                nn.init.normal(self.__getattr__(name), 0.0, 0.01)
                # nn.init.constant(self.__getattr__(name), 1.0)  # different from zhang's 0

    @staticmethod
    def _forward_rnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial, drop_masks):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(list(range(max_time))):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
            c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)  # transpose: return the transpose matrix
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)  # expand: -1 means not expand that dimension
        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)  # h0, c0
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:  # when training, use the dropout
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                # permute: exchange the dimension
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            layer_output, (layer_h_n, layer_c_n) = VariationalLSTM._forward_rnn(cell=self.fcells[layer], \
                                                                                input=input, masks=masks,
                                                                                initial=initial,
                                                                                drop_masks=hidden_mask)
            if self.bidirectional:
                blayer_output, (blayer_h_n, blayer_c_n) = VariationalLSTM._forward_brnn(cell=self.bcells[layer], \
                                                                                        input=input, masks=masks,
                                                                                        initial=initial,
                                                                                        drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        if self.batch_first:
            input = input.transpose(1, 0)  # transpose: return the transpose matrix
        return input, (h_n, c_n)
