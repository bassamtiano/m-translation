import random

import numpy as np
import tensorflow as tf

import tensorflow.python.ops.math_ops import sigmoid
import tensorflow.python.ops.math_ops import tanh

def get_optimizer(opt):

class TransformerModel(object):
    def __init__():

    def init_encoder(self):

    def init_decoder(self):

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


def attention_layer():


def create_transformer_model():

    if hidden_size % num_attention_heads != 0:
        raise ValueError(

        )

    attention_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank)

    if input_width != hidden_size:
        raise ValueError(

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
    

def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is None:
        assert_rank(tensor, expected_rank, name)
