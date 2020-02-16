# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from hanlp.utils.tf_util import hanlp_register


@hanlp_register
class WeightNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```

    Args:
      layer: a layer instance
      data_init: If

    Returns:

    Raises:
      ValueError: If not initialized with a
      ValueError: If
      NotImplementedError: If

    """

    def __init__(self, layer, data_init=True, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name='layer')
        self._init_critical_section = tf.CriticalSection(name='init_mutex')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

    def build(self, input_shape):
        """Build `Layer`

        Args:
          input_shape: 

        Returns:

        """
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`WeightNormalization` must wrap a layer that'
                             ' contains a `kernel` for weights')

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel_layer.kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel_layer.kernel.shape.rank - 1))

        self.g = self.add_weight(
            name='g',
            shape=(self.layer_depth,),
            initializer='ones',
            dtype=kernel_layer.kernel.dtype,
            trainable=True)
        self.v = kernel_layer.kernel

        self._initialized = self.add_weight(
            name='initialized',
            shape=None,
            initializer='zeros',
            dtype=tf.dtypes.bool,
            trainable=False)

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope('data_dep_init'):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config['config']['trainable'] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(
                    layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if self.is_rnn:
                    self._naked_clone_layer.cell.activation = None
                else:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`

        Args:
          inputs: 

        Returns:

        """

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = self._init_critical_section.execute(lambda: tf.cond(
            self._initialized, _do_nothing, _update_weights))

        with tf.name_scope('compute_weights'):
            # Replace kernel by normalized weight variable.
            self.layer.kernel = tf.nn.l2_normalize(
                self.v, axis=self.kernel_norm_axes) * g

            # Ensure we calculate result after updating kernel.
            update_kernel = tf.identity(self.layer.kernel)
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.

        Args:
          inputs: 

        Returns:

        """
        with tf.control_dependencies([
            tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                self._initialized,
                False,
                message='The layer has been initialized.')
        ]):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization.

        Args:
          inputs: 

        Returns:

        """
        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, 'bias') and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {'data_init': self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
