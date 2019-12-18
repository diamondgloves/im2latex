# from __future__ import division
import math
import tensorflow as tf
# from six.moves import xrange

# taken from https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py

def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be expressed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = shape_list(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in range(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x

def add_positional_embedding_nd(x, max_length, name=None):
  """Adds n-dimensional positional embedding.
  The embeddings add to all positional dimensions of the tensor.
  Args:
    x: Tensor with shape [batch, p1 ... pn, depth]. It has n positional
      dimensions, i.e., 1 for text, 2 for images, 3 for video, etc.
    max_length: int representing static maximum size of any dimension.
    name: str representing name of the embedding tf.Variable.
  Returns:
    Tensor of same shape as x.
  """
  with tf.name_scope("add_positional_embedding_nd"):
    x_shape = shape_list(x)
    num_dims = len(x_shape) - 2
    depth = x_shape[-1]
    base_shape = [1] * (num_dims + 1) + [depth]
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [1] * num_dims + [depth]
    for i in range(num_dims):
      shape = base_shape[:]
      start = base_start[:]
      size = base_size[:]
      shape[i + 1] = max_length
      size[i + 1] = x_shape[i + 1]
      var = tf.get_variable(
          name + "_%d" % i,
          shape,
          initializer=tf.random_normal_initializer(0, depth**-0.5))
      var = var * depth**0.5
      x += tf.slice(var, start, size)
    return x

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
