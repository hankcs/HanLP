# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 23:04
import tensorflow as tf
from hanlp.layers.transformers.tf_imports import TFPreTrainedModel
from hanlp.components.parsers.biaffine_tf.layers import IndependentDropout, SharedDropout, Biaffine, MLP


class BiaffineModelTF(tf.keras.Model):

    def __init__(self, config, embed=None, transformer: TFPreTrainedModel = None):
        """An implementation of T. Dozat and C. D. Manning, “Deep Biaffine Attention for Neural Dependency Parsing.,” ICLR, 2017.
            Although I have my MXNet implementation, I found zysite's PyTorch implementation is cleaner so I port it to TensorFlow

        Args:
          config: param embed:

        Returns:

        """
        super(BiaffineModelTF, self).__init__()
        assert not (embed and transformer), 'Either pre-trained word embed and transformer is supported, but not both'
        normal = tf.keras.initializers.RandomNormal(stddev=1.)
        if not transformer:
            # the embedding layer
            self.word_embed = tf.keras.layers.Embedding(input_dim=config.n_words,
                                                        output_dim=config.n_embed,
                                                        embeddings_initializer=tf.keras.initializers.zeros() if embed
                                                        else normal,
                                                        name='word_embed')
            self.feat_embed = tf.keras.layers.Embedding(input_dim=config.n_feats,
                                                        output_dim=config.n_embed,
                                                        embeddings_initializer=tf.keras.initializers.zeros() if embed
                                                        else normal,
                                                        name='feat_embed')
            self.embed_dropout = IndependentDropout(p=config.embed_dropout, name='embed_dropout')

            # the word-lstm layer
            self.lstm = tf.keras.models.Sequential(name='lstm')
            for _ in range(config.n_lstm_layers):
                self.lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    units=config.n_lstm_hidden,
                    dropout=config.lstm_dropout,
                    recurrent_dropout=config.lstm_dropout,
                    return_sequences=True,
                    kernel_initializer='orthogonal',
                    unit_forget_bias=False,  # turns out to hinder performance
                )))
            self.lstm_dropout = SharedDropout(p=config.lstm_dropout, name='lstm_dropout')
        else:
            self.transformer = transformer
            transformer_dropout = config.get('transformer_dropout', None)
            if transformer_dropout:
                self.transformer_dropout = SharedDropout(p=config.transformer_dropout, name='transformer_dropout')
            d_positional = config.get('d_positional', None)
            if d_positional:
                max_seq_length = config.get('max_seq_length', 256)
                self.position_table = self.add_weight(shape=(max_seq_length, d_positional),
                                                      initializer='random_normal',
                                                      trainable=True)
        # the MLP layers
        self.mlp_arc_h = MLP(n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout, name='mlp_arc_h')
        self.mlp_arc_d = MLP(n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout, name='mlp_arc_d')
        self.mlp_rel_h = MLP(n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout, name='mlp_rel_h')
        self.mlp_rel_d = MLP(n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout, name='mlp_rel_d')

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False, name='arc_attn')
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True, name='rel_attn')
        if embed is not None:
            self.pretrained = embed
        self.pad_index = tf.constant(config.pad_index, dtype=tf.int64)
        self.unk_index = tf.constant(config.unk_index, dtype=tf.int64)

    # noinspection PyMethodOverriding
    def call(self, inputs, mask_inf=True, **kwargs):
        # batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        # mask = words.ne(self.pad_index)
        if hasattr(self, 'lstm'):
            words, feats = inputs
            mask = tf.not_equal(words, self.pad_index)
            # set the indices larger than num_embeddings to unk_index
            # ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_mask = tf.greater_equal(words, self.word_embed.input_dim)
            ext_words = tf.where(ext_mask, self.unk_index, words)

            # get outputs from embedding layers
            word_embed = self.word_embed(ext_words)
            if hasattr(self, 'pretrained'):
                word_embed += self.pretrained(words)
            feat_embed = self.feat_embed(feats)
            word_embed, feat_embed = self.embed_dropout([word_embed, feat_embed])
            # concatenate the word and feat representations
            embed = tf.concat((word_embed, feat_embed), axis=-1)

            x = self.lstm(embed, mask=mask)
            x = self.lstm_dropout(x)
        else:
            words, (input_ids, input_mask, prefix_offset) = inputs
            mask = tf.not_equal(words, self.pad_index)
            x = self.run_transformer(input_ids, input_mask, prefix_offset)

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = tf.transpose(self.rel_attn(rel_d, rel_h), [0, 2, 3, 1])
        # set the scores that exceed the length of each sentence to -inf
        if mask_inf:
            s_arc = tf.where(tf.expand_dims(mask, 1), s_arc, float('-inf'))

        return s_arc, s_rel

    def run_transformer(self, input_ids, input_mask, prefix_offset):
        if isinstance(self.transformer, TFPreTrainedModel):
            sequence_output = self.transformer([input_ids, input_mask])
            sequence_output = sequence_output[0]
        else:
            sequence_output = self.transformer([input_ids, tf.zeros_like(input_ids)], mask=input_mask)
        x = tf.gather(sequence_output, prefix_offset, batch_dims=1)
        if hasattr(self, 'transformer_dropout'):
            x = self.transformer_dropout(x)
        if hasattr(self, 'position_table'):
            batch_size, seq_length = tf.shape(x)[:2]
            timing_signal = tf.broadcast_to(self.position_table[:seq_length],
                                            [batch_size, seq_length, self.position_table.shape[-1]])
            x = tf.concat([x, timing_signal], axis=-1)
        return x

    def to_functional(self):
        words = tf.keras.Input(shape=[None], dtype=tf.int64, name='words')
        feats = tf.keras.Input(shape=[None], dtype=tf.int64, name='feats')
        s_arc, s_rel = self.call([words, feats], mask_inf=False)
        return tf.keras.Model(inputs=[words, feats], outputs=[s_arc, s_rel])
