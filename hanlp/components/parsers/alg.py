# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 19:49
# Ported from the PyTorch implementation https://github.com/zysite/biaffine-parser
from typing import List

import tensorflow as tf


def nonzero(t: tf.Tensor) -> tf.Tensor:
    return tf.where(t > 0)


def view(t: tf.Tensor, *dims) -> tf.Tensor:
    return tf.reshape(t, dims)


def arange(n: int) -> tf.Tensor:
    return tf.range(n)


def randperm(n: int) -> tf.Tensor:
    return tf.random.shuffle(arange(n))


def tolist(t: tf.Tensor) -> List:
    if isinstance(t, tf.Tensor):
        t = t.numpy()
    return t.tolist()


def kmeans(x, k):
    """
    See https://github.com/zysite/biaffine-parser/blob/master/parser/utils/alg.py#L7
    :param x:
    :param k:
    :return:
    """
    x = tf.constant(x, dtype=tf.float32)
    # count the frequency of each datapoint
    d, indices, f = tf.unique_with_counts(x, tf.int32)
    f = tf.cast(f, tf.float32)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = tf.random.shuffle(d)[:k], None
    # assign labels to each datapoint based on centroids
    dists = tf.abs(tf.expand_dims(d, -1) - c)
    y = tf.argmin(dists, axis=-1, output_type=tf.int32)
    dists = tf.gather_nd(dists, tf.transpose(tf.stack([tf.range(tf.shape(dists)[0], dtype=tf.int32), y])))
    # make sure number of datapoints is greater than that of clusters
    assert len(x) >= k, f"unable to assign {len(x)} datapoints to {k} clusters"

    while old is None or not tf.reduce_all(c == old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not tf.reduce_any(y == i):
                mask = tf.cast(y == tf.expand_dims(tf.range(k, dtype=tf.int32), -1), tf.float32)
                lens = tf.reduce_sum(mask, axis=-1)
                biggest = view(nonzero(mask[tf.argmax(lens)]), -1)
                farthest = tf.argmax(tf.gather(dists, biggest))
                tf.tensor_scatter_nd_update(y, tf.expand_dims(tf.expand_dims(biggest[farthest], -1), -1), [i])
        mask = tf.cast(y == tf.expand_dims(tf.range(k, dtype=tf.int32), -1), tf.float32)
        # update the centroids
        c, old = tf.cast(tf.reduce_sum(total * mask, axis=-1), tf.float32) / tf.cast(tf.reduce_sum(f * mask, axis=-1),
                                                                                     tf.float32), c
        # re-assign all datapoints to clusters
        dists = tf.abs(tf.expand_dims(d, -1) - c)
        y = tf.argmin(dists, axis=-1, output_type=tf.int32)
        dists = tf.gather_nd(dists, tf.transpose(tf.stack([tf.range(tf.shape(dists)[0], dtype=tf.int32), y])))
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, (assigned, _) = tf.gather(y, indices), tf.unique(y)
    # get the centroids of the assigned clusters
    centroids = tf.gather(c, assigned).numpy().tolist()
    # map all values of datapoints to buckets
    clusters = [tf.squeeze(tf.where(y == i), axis=-1).numpy().tolist() for i in assigned]

    return centroids, clusters
