# Adopted from https://github.com/KiroSummer/A_Syntax-aware_MTL_Framework_for_Chinese_SRL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from .layer import DropoutLayer, HighwayLSTMCell, VariationalLSTMCell


def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)


class HighwayBiLSTM(nn.Module):
    """A module that runs multiple steps of HighwayBiLSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=False, dropout_in=0,
                 dropout_out=0):
        super(HighwayBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells, self.f_dropout, self.f_hidden_dropout = [], [], []
        self.bcells, self.b_dropout, self.b_hidden_dropout = [], [], []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.fcells.append(HighwayLSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            self.f_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.f_hidden_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            if self.bidirectional:
                self.bcells.append(HighwayLSTMCell(input_size=hidden_size, hidden_size=hidden_size))
                self.b_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
                self.b_hidden_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
        self.fcells, self.bcells = nn.ModuleList(self.fcells), nn.ModuleList(self.bcells)
        self.f_dropout, self.b_dropout = nn.ModuleList(self.f_dropout), nn.ModuleList(self.b_dropout)

    def reset_dropout_layer(self, batch_size):
        for layer in range(self.num_layers):
            self.f_dropout[layer].reset_dropout_mask(batch_size)
            if self.bidirectional:
                self.b_dropout[layer].reset_dropout_mask(batch_size)

    @staticmethod
    def _forward_rnn(cell, gate, input, masks, initial, drop_masks=None, hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input[time], mask=masks[time], hx=hx, dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, gate, input, masks, initial, drop_masks=None, hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(list(range(max_time))):
            h_next, c_next = cell(input[time], mask=masks[time], hx=hx, dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)  # transpose: return the transpose matrix
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()

        self.reset_dropout_layer(batch_size)  # reset the dropout each batch forward

        masks = masks.expand(-1, -1, self.hidden_size)  # expand: -1 means not expand that dimension
        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)  # h0, c0

        h_n, c_n = [], []
        for layer in range(self.num_layers):
            # hidden_mask, hidden_drop = None, None
            hidden_mask, hidden_drop = self.f_dropout[layer], self.f_hidden_dropout[layer]
            layer_output, (layer_h_n, layer_c_n) = HighwayBiLSTM._forward_rnn(cell=self.fcells[layer], \
                                                                              gate=None, input=input, masks=masks,
                                                                              initial=initial, \
                                                                              drop_masks=hidden_mask,
                                                                              hidden_drop=hidden_drop)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
            if self.bidirectional:
                hidden_mask, hidden_drop = self.b_dropout[layer], self.b_hidden_dropout[layer]
                blayer_output, (blayer_h_n, blayer_c_n) = HighwayBiLSTM._forward_brnn(cell=self.bcells[layer], \
                                                                                      gate=None, input=layer_output,
                                                                                      masks=masks, initial=initial, \
                                                                                      drop_masks=hidden_mask,
                                                                                      hidden_drop=hidden_drop)
                h_n.append(blayer_h_n)
                c_n.append(blayer_c_n)

            input = blayer_output if self.bidirectional else layer_output

        h_n, c_n = torch.stack(h_n, 0), torch.stack(c_n, 0)
        if self.batch_first:
            input = input.transpose(1, 0)  # transpose: return the transpose matrix
        return input, (h_n, c_n)


class StackedHighwayBiLSTM(nn.Module):
    """A module that runs multiple steps of HighwayBiLSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(StackedHighwayBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1

        self.fcells, self.f_dropout, self.f_hidden_dropout = [], [], []
        self.bcells, self.b_dropout, self.b_hidden_dropout = [], [], []
        self.f_initial, self.b_initial = [], []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else 2 * hidden_size if self.bidirectional else hidden_size
            self.fcells.append(VariationalLSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            self.f_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.f_hidden_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.f_initial.append(nn.Parameter(torch.Tensor(2, self.hidden_size)))
            assert self.bidirectional is True
            self.bcells.append(VariationalLSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
            self.b_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.b_hidden_dropout.append(DropoutLayer(hidden_size, self.dropout_out))
            self.b_initial.append(nn.Parameter(torch.Tensor(2, self.hidden_size)))
        self.lstm_project_layer = nn.ModuleList([nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
                                                 for _ in range(num_layers - 1)])
        self.fcells, self.bcells = nn.ModuleList(self.fcells), nn.ModuleList(self.bcells)
        self.f_dropout, self.b_dropout = nn.ModuleList(self.f_dropout), nn.ModuleList(self.b_dropout)
        self.f_hidden_dropout, self.b_hidden_dropout = \
            nn.ModuleList(self.f_hidden_dropout), nn.ModuleList(self.b_hidden_dropout)
        self.f_initial, self.b_initial = nn.ParameterList(self.f_initial), nn.ParameterList(self.b_initial)
        self.reset_parameters()

    def reset_parameters(self):
        for layer_initial in [self.f_initial, self.b_initial]:
            for initial in layer_initial:
                init.xavier_uniform_(initial)
        for layer in self.lstm_project_layer:
            init.xavier_uniform_(layer.weight)
            initializer_1d(layer.bias, init.xavier_uniform_)

    def reset_dropout_layer(self, batch_size):
        for layer in range(self.num_layers):
            self.f_dropout[layer].reset_dropout_mask(batch_size)
            self.f_hidden_dropout[layer].reset_dropout_mask(batch_size)
            if self.bidirectional:
                self.b_dropout[layer].reset_dropout_mask(batch_size)
                self.b_hidden_dropout[layer].reset_dropout_mask(batch_size)

    def reset_state(self, batch_size):
        f_states, b_states = [], []
        for f_layer_initial, b_layer_initial in zip(self.f_initial, self.b_initial):
            f_states.append([f_layer_initial[0].expand(batch_size, -1), f_layer_initial[1].expand(batch_size, -1)])
            b_states.append([b_layer_initial[0].expand(batch_size, -1), b_layer_initial[1].expand(batch_size, -1)])
        return f_states, b_states

    @staticmethod
    def _forward_rnn(cell, gate, input, masks, initial, drop_masks=None, hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input[time], mask=masks[time], hx=hx, dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, gate, input, masks, initial, drop_masks=None, hidden_drop=None):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(list(range(max_time))):
            h_next, c_next = cell(input[time], mask=masks[time], hx=hx, dropout=drop_masks)
            hx = (h_next, c_next)
            output.append(h_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)  # transpose: return the transpose matrix
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()

        self.reset_dropout_layer(batch_size)  # reset the dropout each batch forward
        f_states, b_states = self.reset_state(batch_size)

        masks = masks.expand(-1, -1, self.hidden_size)  # expand: -1 means not expand that dimension

        h_n, c_n = [], []
        outputs = []
        for layer in range(self.num_layers):
            hidden_mask, hidden_drop = self.f_dropout[layer], self.f_hidden_dropout[layer]
            layer_output, (layer_h_n, layer_c_n) = \
                StackedHighwayBiLSTM._forward_rnn(cell=self.fcells[layer],
                                                  gate=None, input=input, masks=masks, initial=f_states[layer],
                                                  drop_masks=hidden_mask, hidden_drop=hidden_drop)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
            assert self.bidirectional is True
            hidden_mask, hidden_drop = self.b_dropout[layer], self.b_hidden_dropout[layer]
            blayer_output, (blayer_h_n, blayer_c_n) = \
                StackedHighwayBiLSTM._forward_brnn(cell=self.bcells[layer],
                                                   gate=None, input=input, masks=masks, initial=b_states[layer],
                                                   drop_masks=hidden_mask, hidden_drop=hidden_drop)
            h_n.append(blayer_h_n)
            c_n.append(blayer_c_n)

            output = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output
            output = F.dropout(output, self.dropout_out, self.training)
            if layer > 0:  # Highway
                highway_gates = torch.sigmoid(self.lstm_project_layer[layer - 1].forward(output))
                output = highway_gates * output + (1 - highway_gates) * input
            if self.batch_first:
                outputs.append(output.transpose(1, 0))
            else:
                outputs.append(output)
            input = output

        h_n, c_n = torch.stack(h_n, 0), torch.stack(c_n, 0)
        if self.batch_first:
            output = output.transpose(1, 0)  # transpose: return the transpose matrix
        return output, (h_n, c_n), outputs
