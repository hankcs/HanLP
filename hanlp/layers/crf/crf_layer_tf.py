# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import tensorflow as tf

from hanlp.layers.crf.crf_tf import crf_decode, crf_log_likelihood


class CRF(tf.keras.layers.Layer):
    """Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).
    
    Note: the loss and accuracy functions of networks using `CRF` must
    use the provided loss and accuracy functions (denoted as loss and viterbi_accuracy)
    as the classification of sequences are used with the layers internal weights.
    
    Copyright: this is a modified version of
    https://github.com/NervanaSystems/nlp-architect/blob/master/nlp_architect/nn/tensorflow/python/keras/layers/crf.py

    Args:
      num_labels(int): the number of labels to tag each temporal input.
    Input shape:
      num_labels(int): the number of labels to tag each temporal input.
    Input shape:
    nD tensor with shape `(batch_size, sentence length, num_classes)`.
    Output shape:
      nD tensor with shape: `(batch_size, sentence length, num_classes)`.

    Returns:

    """

    def __init__(self, num_classes, **kwargs):
        self.transitions = None
        super(CRF, self).__init__(**kwargs)
        # num of output labels
        self.output_dim = int(num_classes)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = False
        sequence_lengths = None

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'supports_masking': self.supports_masking,
            'transitions': tf.keras.backend.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    # pylint: disable=arguments-differ
    def call(self, inputs, sequence_lengths=None, mask=None, training=None, **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            sequence_lengths = tf.keras.backend.flatten(sequence_lengths)
        else:
            sequence_lengths = tf.math.count_nonzero(mask, axis=1)

        viterbi_sequence, _ = crf_decode(sequences, self.transitions,
                                         sequence_lengths)
        output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
        return tf.keras.backend.in_train_phase(sequences, output)

    # def loss(self, y_true, y_pred):
    #     y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
    #     log_likelihood, self.transitions = \
    #         crf_log_likelihood(y_pred,
    #                            tf.cast(y_true, dtype=tf.int32),
    #                            sequence_lengths,
    #                            transition_params=self.transitions)
    #     return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    @property
    def viterbi_accuracy(self):
        def accuracy(y_true, y_pred):
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            viterbi_sequence, _ = crf_decode(y_pred, self.transitions,
                                             sequence_lengths)
            output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
            return tf.keras.metrics.categorical_accuracy(y_true, output)

        accuracy.func_name = 'viterbi_accuracy'
        return accuracy


class CRFLoss(object):

    def __init__(self, crf: CRF, dtype) -> None:
        super().__init__()
        self.crf = crf
        self.dtype = dtype
        self.__name__ = type(self).__name__

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        assert sample_weight is not None, 'your model has to support masking'
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        sequence_lengths = tf.math.count_nonzero(sample_weight, axis=1)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        log_likelihood, self.crf.transitions = \
            crf_log_likelihood(y_pred,
                               tf.cast(y_true, dtype=tf.int32),
                               sequence_lengths,
                               transition_params=self.crf.transitions)
        return tf.reduce_mean(-log_likelihood)


class CRFWrapper(tf.keras.Model):
    def __init__(self, model: tf.keras.Model, num_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.crf = CRF(model.output.shape[-1] if not num_classes else num_classes)

    def call(self, inputs, training=None, mask=None):
        output = self.model(inputs, training=training, mask=mask)
        viterbi_output = self.crf(output)
        return viterbi_output

    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)
