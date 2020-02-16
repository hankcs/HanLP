# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-05 19:34
from alnlp.modules.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hanlp.common.structure import ConfigTracker


class _LSTMSeq2Seq(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            dropout: float = 0.0,
            bidirectional: bool = False,
    ):
        """
        Under construction, not ready for production
        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param bias:
        :param dropout:
        :param bidirectional:
        """
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, embed, lens, max_len):
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, True, total_length=max_len)
        return x


# We might update this to support yaml based configuration
class LSTMContextualEncoder(LstmSeq2SeqEncoder, ConfigTracker):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, dropout: float = 0.0,
                 bidirectional: bool = False, stateful: bool = False):
        super().__init__(input_size, hidden_size, num_layers, bias, dropout, bidirectional, stateful)
        ConfigTracker.__init__(self, locals())
