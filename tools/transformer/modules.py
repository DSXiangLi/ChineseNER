# -*-coding:utf-8 -*-
"""
    Transformer modules for the original paper: Attention is all you need
"""
import tensorflow as tf
import numpy as np

from tools.utils import add_layer_summary


def embedding_project(embedding, d_model):
    # emb_size is the raw pretrain char / bichar embedding size
    emb_size = embedding.shape.as_list()[-1] # emb_size
    if emb_size == d_model:
        return embedding
    else:
        # do linear transformation, mainly to enlarge the input embedding space
        embedding = tf.layers.dense(embedding, units=d_model, activation=None, use_bias=True)
        add_layer_summary('embedding_project', embedding)
    return embedding


def seq_mask_gen(seq_len, max_seq_len):
    """
    Default paddinng mask for sequence
    input:
        input_: {'tokens':, 'seq_len'}, seq_len is non-padded sequence len
    output:
        mask: batch * 1 * key_len, with 1 to keep,0 to drop
    """
    mask = tf.sequence_mask(lengths=tf.to_int32(seq_len), maxlen=max_seq_len,
                            dtype=tf.float32)

    # add axis1 to enable broadcast to query_len * key_len later in scaled-dot
    mask = tf.expand_dims(mask, axis=1)

    return mask


def layer_norm(x):
    """
    layer normalization from Jimmy, apply normalization along feature and apply transformation
    """
    with tf.variable_scope('layer_normalization', reuse=tf.AUTO_REUSE):
        d_model = x.shape.as_list()[-1]
        epsilon = tf.constant(np.finfo(np.float32).eps)
        mean, variance = tf.nn.moments(x, axes=-1, keep_dims=True)
        x = (x - mean)/((variance + epsilon)**0.5) # do layer norm
        add_layer_summary('norm', x)

        kernel = tf.get_variable('norm_kernel', shape=(d_model,), initializer=tf.ones_initializer())
        bias = tf.get_variable('norm_bias', shape=(d_model,), initializer=tf.zeros_initializer())
        x= tf.multiply(kernel, x) + bias
        add_layer_summary('norm_transform', x)
    return x


def add_and_norm_layer(x, sub_layer_x):
    """
    combine Residual connection & layer_norm
    """
    with tf.variable_scope('add_and_norm'):
        x = tf.add(x, sub_layer_x)
        x = layer_norm(x)
    return x


def ffn(x, ffn_hidden, dropout_rate, is_training):
    """
    feed forward + add & norm
    """
    with tf.variable_scope('ffn', reuse=tf.AUTO_REUSE):
        d_model = x.shape.as_list()[-1]  # emb_size
        y = tf.layers.dense(x, units=ffn_hidden, activation='relu', name='ffn_inner')
        add_layer_summary('ffn_hidden1', y)
        y = tf.layers.dense(y, units=d_model, activation=None, name='ffn_outer')
        add_layer_summary('ffn_hidden2', y)
        y = tf.layers.dropout(y, rate=dropout_rate, training=is_training)
        y = add_and_norm_layer(x, y)
    return y


def future_mask_gen(seq_len, max_seq_len):
    """
    In decoder self-attention, additional future_mask is needed to avoid leakage. add future mask on padding mask
    input:
        input_: {'tokens':, 'seq_len'}, seq_len is non-padded sequence len
    output:
        mask:  batch_size * input_len * input_len, with 1 to keep, 0 to drop
    """
    # give 0 to all padding position for both key and query
    seq_mask = seq_mask_gen(seq_len, max_seq_len) # batch_size * 1 * key_len
    # batch_size * key_len * key_len(seq_len)
    mask = tf.matmul(seq_mask, seq_mask, transpose_a=True)
    # keep lower triangle with diagonal
    mask = tf.matrix_band_part(mask, num_lower=-1, num_upper=0)

    return mask


def scaled_dot_product_attention(key, query):
    """
    apply dot product attention with mask
    input:
        key: (batch_size*num_head) * key_len * emb_size
        query: (batch_size*num_head) * query_len * emb_size
    output:
        weighted_val: unnormalized attention (batch_size*num_head) * query_len * emb_size
    """
    with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
        # scalaed weight matrix : batch_size * query_len * key_len
        dk = tf.cast(key.shape.as_list()[-1], tf.float32)# emb_size
        weight = tf.matmul(query, key, transpose_b=True)/(dk**0.5)
        add_layer_summary('att_premask', weight)
    return weight


def normalize_attention(weight, mask):
    """
    apply mask on attention and normalize on key dimension
    Input:
        weight: (batch_size * num_head) * query_len * key_len
        mask: (batch_size * num_head) * 1 * key_len
    """
    # apply mask: large negative will become 0 in softmax[mask=0 ignore]
    weight += (1 - mask) * (-2 ** 32 + 1)
    weight = tf.nn.softmax(weight, axis=-1)
    tf.summary.image("attention", tf.expand_dims(weight[:1], -1))  # add channel dim for visiualization
    add_layer_summary('attention', weight)
    return weight


def multi_head_attention(key, value, query, mask, num_head, dropout_rate, is_training):
    """
    Mutlihead attention with mask
    input:
        key: batch_size * key_len * emb_size
        query: batch_size * query_len * emb_size
        value: batch_size * key_len * emb_size
        mask: batch_size * key_len
        attention_function: absolute attention / relative attention
    output:
        weighted_val: batch_size * query_len * emb_size
    """
    with tf.variable_scope('multi_head_attention', reuse=tf.AUTO_REUSE):
        d_model = value.shape.as_list()[-1] # emb_size
        # linear projection with dimension unchaangned
        new_key = tf.layers.dense(key, units=d_model, activation=None, name='pre_key_project') # batch_size * key_len * emb_size
        new_value = tf.layers.dense(value, units=d_model, activation=None, name='pre_value_project')
        new_query = tf.layers.dense(query, units=d_model, activation=None, name='pre_query_project')

        # split d_model by num_head and compute attention in parallel
        # (batch_size * num_head) * key_len * (emb_size/num_head)
        new_key = tf.concat(tf.split(new_key, num_or_size_splits=num_head, axis=-1), axis=0)
        new_value = tf.concat(tf.split(new_value, num_or_size_splits=num_head, axis=-1), axis=0)
        new_query = tf.concat(tf.split(new_query, num_or_size_splits=num_head, axis=-1), axis=0)

        # calculate dot-product attention
        weight = scaled_dot_product_attention(new_key, new_query)
        weight = normalize_attention(weight, tf.tile(mask, [num_head, 1, 1]))

        # weighted value & concat num_head back
        # (batch_size * num_head) * query_len * (emb_size/num_head) -> batch_size * query_len * emb_size
        weighted_val = tf.matmul(weight, new_value)
        weighted_val = tf.concat(tf.split(weighted_val, num_or_size_splits=num_head, axis=0), axis=-1)

        # Linear projection
        weighted_val = tf.layers.dense(weighted_val, units=d_model, activation=None, name='post_linear_project')
        add_layer_summary('raw_multi_head', weighted_val)
        # Do dropout + add&norm
        weighted_val = tf.layers.dropout(weighted_val, rate=dropout_rate, training=is_training)
        weighted_val = add_and_norm_layer(query, weighted_val)

    return weighted_val


def sinusoidal_positional_encoding(emb_dim, pos_seq):
    """
    inject relative & absolute position information
    inputs:
        emb_dim: embedding dimension of positional encoding
        pos_seq: positional sequence.can be [-max_seq_len,...max_seq_len), [0, ..max_seq_len], [0,...clip]
    output:
        encoding: 2*max_seq_len * emb_size[-max_seq_len, ...-1,0, 1,...max_seq_len-1]
    """
    with tf.variable_scope('positional_encoding'):
        inv_freq = np.array([1/(10000 ** ((i - i % 2) /emb_dim)) for i in range(emb_dim)])
        encoding_matrix = np.einsum('i,j->ij', pos_seq, inv_freq) # max_seq * emb_dim

        def sin_cos(row):
            #部分实现是[sin,cos,sin,cos...],部分是[sin,sin....cos,cos] 这两者是permuatation invariant,不用纠结
            row = [np.cos(val) if i % 2 else np.sin(val) for i, val in enumerate(row)]
            return row
        encoding_matrix = np.apply_along_axis(sin_cos, 1, encoding_matrix)
        encoding_matrix = tf.cast(tf.constant(encoding_matrix), tf.float32)

    return encoding_matrix


def get_pos_embedding(input_ids, pos_encoding, max_seq_len):
    # not using batch_size in params, because whether drop remainder can affect batch_size
    batch_size = tf.shape(input_ids)[0]
    pos_id = tf.tile(tf.expand_dims(tf.range(max_seq_len), 0),
                     [batch_size, 1])  # batch_size * max_seq_len
    pe = tf.nn.embedding_lookup(pos_encoding, pos_id) # batch_size * max_seq_len * emb_dim
    return pe


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)