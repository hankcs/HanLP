# Adopted from https://github.com/allenai/allennlp under Apache Licence 2.0.
# Changed the packaging and created a subclass CharCNNEmbedding

from typing import Union, Tuple, Optional, Callable
import torch
from torch import nn
from alnlp.modules.cnn_encoder import CnnEncoder
from alnlp.modules.time_distributed import TimeDistributed
from hanlp_common.configurable import AutoConfigurable
from hanlp.common.transform import VocabDict, ToChar
from hanlp.common.vocab import Vocab
from hanlp.layers.embeddings.embedding import EmbeddingDim, Embedding


class CharCNN(nn.Module):
    def __init__(self,
                 field: str,
                 embed: Union[int, Embedding], num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
                 conv_layer_activation: str = 'ReLU',
                 output_dim: Optional[int] = None,
                 vocab_size=None) -> None:
        """A `CnnEncoder` is a combination of multiple convolution layers and max pooling layers.
        The input to this module is of shape `(batch_size, num_tokens,
        input_dim)`, and the output is of shape `(batch_size, output_dim)`.

        The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
        out a vector of size num_filters. The number of times a convolution layer will be used
        is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
        outputs from the convolution layer and outputs the max.

        This operation is repeated for every ngram size passed, and consequently the dimensionality of
        the output after maxpooling is `len(ngram_filter_sizes) * num_filters`.  This then gets
        (optionally) projected down to a lower dimensional output, specified by `output_dim`.

        We then use a fully connected layer to project in back to the desired output_dim.  For more
        details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
        Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

        See allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder, Apache 2.0

        Args:
            field: The field in samples this encoder will work on.
            embed: An ``Embedding`` object or the feature size to create an ``Embedding`` object.
            num_filters: This is the output dim for each convolutional layer, which is the number of "filters"
                learned by that layer.
            ngram_filter_sizes: This specifies both the number of convolutional layers we will create and their sizes.  The
                default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
                ngrams of size 2 to 5 with some number of filters.
            conv_layer_activation: `Activation`, optional (default=`torch.nn.ReLU`)
                Activation to use after the convolution layers.
            output_dim: After doing convolutions and pooling, we'll project the collected features into a vector of
                this size.  If this value is `None`, we will just return the result of the max pooling,
                giving an output of shape `len(ngram_filter_sizes) * num_filters`.
            vocab_size: The size of character vocab.

        Returns:
            A tensor of shape `(batch_size, output_dim)`.
        """
        super().__init__()
        EmbeddingDim.__init__(self)
        # the embedding layer
        if isinstance(embed, int):
            embed = nn.Embedding(num_embeddings=vocab_size,
                                 embedding_dim=embed)
        else:
            raise ValueError(f'Unrecognized type for {embed}')
        self.field = field
        self.embed = TimeDistributed(embed)
        self.encoder = TimeDistributed(
            CnnEncoder(embed.embedding_dim, num_filters, ngram_filter_sizes, conv_layer_activation, output_dim))
        self.embedding_dim = output_dim or num_filters * len(ngram_filter_sizes)

    def forward(self, batch: dict, **kwargs):
        tokens: torch.Tensor = batch[f'{self.field}_char_id']
        mask = tokens.ge(0)
        x = self.embed(tokens)
        return self.encoder(x, mask)

    def get_output_dim(self) -> int:
        return self.embedding_dim


class CharCNNEmbedding(Embedding, AutoConfigurable):
    def __init__(self,
                 field,
                 embed: Union[int, Embedding],
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
                 conv_layer_activation: str = 'ReLU',
                 output_dim: Optional[int] = None,
                 min_word_length=None
                 ) -> None:
        """

        Args:
            field: The character field in samples this encoder will work on.
            embed: An ``Embedding`` object or the feature size to create an ``Embedding`` object.
            num_filters: This is the output dim for each convolutional layer, which is the number of "filters"
                learned by that layer.
            ngram_filter_sizes: This specifies both the number of convolutional layers we will create and their sizes.  The
                default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
                ngrams of size 2 to 5 with some number of filters.
            conv_layer_activation: `Activation`, optional (default=`torch.nn.ReLU`)
                Activation to use after the convolution layers.
            output_dim: After doing convolutions and pooling, we'll project the collected features into a vector of
                this size.  If this value is `None`, we will just return the result of the max pooling,
                giving an output of shape `len(ngram_filter_sizes) * num_filters`.
            min_word_length: For ngram filter with max size, the input (chars) is required to have at least max size
                chars.
        """
        super().__init__()
        if min_word_length is None:
            min_word_length = max(ngram_filter_sizes)
        self.min_word_length = min_word_length
        self.output_dim = output_dim
        self.conv_layer_activation = conv_layer_activation
        self.ngram_filter_sizes = ngram_filter_sizes
        self.num_filters = num_filters
        self.embed = embed
        self.field = field

    def transform(self, vocabs: VocabDict, **kwargs) -> Optional[Callable]:
        if isinstance(self.embed, Embedding):
            self.embed.transform(vocabs=vocabs)
        vocab_name = self.vocab_name
        if vocab_name not in vocabs:
            vocabs[vocab_name] = Vocab()
        return ToChar(self.field, vocab_name, min_word_length=self.min_word_length,
                      pad=vocabs[vocab_name].safe_pad_token)

    @property
    def vocab_name(self):
        vocab_name = f'{self.field}_char'
        return vocab_name

    def module(self, vocabs: VocabDict, **kwargs) -> Optional[nn.Module]:
        embed = self.embed
        if isinstance(embed, Embedding):
            embed = embed.module(vocabs=vocabs)
        return CharCNN(self.field,
                       embed,
                       self.num_filters,
                       self.ngram_filter_sizes,
                       self.conv_layer_activation,
                       self.output_dim,
                       vocab_size=len(vocabs[self.vocab_name]))
