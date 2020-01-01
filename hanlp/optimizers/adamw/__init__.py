# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-11 18:44
import tensorflow as tf
from hanlp.optimizers.adamw.optimization import WarmUp, AdamWeightDecay


# from hanlp.optimization.adamw.optimizers_v2 import AdamW
# from hanlp.optimization.adamw.utils import get_weight_decays


# def create_optimizer(model, init_lr, num_train_steps, num_warmup_steps):
#     """Creates an optimizer with learning rate schedule."""
#     wd_dict = get_weight_decays(model)
#
#     # Implements linear decay of the learning rate.
#     learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#         initial_learning_rate=init_lr,
#         decay_steps=num_train_steps,
#         end_learning_rate=0.0)
#     if num_warmup_steps:
#         learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
#                                   decay_schedule_fn=learning_rate_fn,
#                                   warmup_steps=num_warmup_steps)
#     optimizer = AdamW(
#         learning_rate=learning_rate_fn,
#         weight_decay_rate=0.01,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-6,
#         exclude_from_weight_decay=['layer_norm', 'bias'])
#     return optimizer


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, weight_decay_rate=0.01, epsilon=1e-6, clipnorm=None):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=0.0)
    if num_warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                                  decay_schedule_fn=learning_rate_fn,
                                  warmup_steps=num_warmup_steps)
    additional_args = {}
    if clipnorm:
        additional_args['clipnorm'] = clipnorm
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=epsilon,
        exclude_from_weight_decay=['LayerNorm', 'bias'],
        **additional_args
    )
    # {'LayerNorm/gamma:0', 'LayerNorm/beta:0'}
    return optimizer
