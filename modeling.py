import random

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class TransformerModel(object):


    def encode():
        with tf.variable_scope("encoder", reuse=tf.AUTO_REFUSE):
            x, seqlens, sentsl = xs

            # embedding

            enc = tf.nn.embedding_lookup(self.embedding_lookup, x)

            # Transformer perform sum residuals on all layers
            if (input_width != hidden_size):
                raise ValueError()

            reshape_to_matrix(input_tensor)
            all_layer_outputs = []

            for layer_idx in range (num_hidden_layers):
                with tf.variable_scope("layer_%d")




    def decode():


def attention_layer(form_tensor,
                    to_tensor,
                    attention_mask = None,
                    num_attention_heads = 1
                    size_per_head=512
                    query_act=None
                    key_act=None
                     ):
    def transpose_for_scores():
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(

        )

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[2]

    elif len(from_shape) == 2:
        if(batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                ""
            )


def create_transformer_model():

    # Confirm the Hidden Size

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden Size (%d) is not a multiple of the number of attention"
            "heads (%d)" % (hidden_size, num_attention_heads)
        )

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    if input_width != hidden_size:
        raise ValueError(
            "The width of the input tensor (%d) != hidden_size (%d)" %
            (input_width, hidden_size)
        )


    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("attention"):
            attention_heads = []
            with tf.variable_scope("self"):
                attention_head = attention_layer(

                )
                attention_head.append(attention_head)

            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)

            with tf.variable_scope("output"):
                attention_output = tf.layers.dense(
                    attention_output,
                    hidden_size,
                    kernel_initialize = create_initializer(initializer_range)
                )
                attention_output = dropout(attention_input, hidden_dropout_prob)
                attention_output = layer_norm(attention_output + layer_input)

            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initialize=create_initializer(initializer_range)


                )

            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initialize = create_initializer(initializer_range)
                )
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is None:
        assert_rank(tensor, expected_rank, name)

def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("At least Rank 2")
    if ndims == 2:
        return input_tensor
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])

def assert_rank(tensor, expected_rank, name=None):
