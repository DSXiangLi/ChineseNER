# -*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tools.transformer.modules import  add_and_norm_layer, normalize_attention, sinusoidal_positional_encoding, add_layer_summary


def four_pos_fusion(pos_s, pos_e, d_model, max_raw_seq_len):
    # generate same relative embedding as TENER, range from [-max_seq_len, max_seq_len]
    pos_seq = np.array(range(-max_raw_seq_len, max_raw_seq_len+1))
    pos_emb = sinusoidal_positional_encoding(d_model, pos_seq)  # (2*max_seq_len) * d_model

    # batch_size * max_seq_len * max_seq_len(max_seq_len = max_char_len + max_lex_len)
    pos_ss = tf.expand_dims(pos_s, -1) - tf.expand_dims(pos_s, -2) + max_raw_seq_len  # add max_raw_seq_len for embedding lookup
    pos_se = tf.expand_dims(pos_s, -1) - tf.expand_dims(pos_e, -2) + max_raw_seq_len
    pos_ee = tf.expand_dims(pos_e, -1) - tf.expand_dims(pos_e, -2) + max_raw_seq_len
    pos_es = tf.expand_dims(pos_e, -1) - tf.expand_dims(pos_s, -2) + max_raw_seq_len

    tf.summary.image("pos_ss", tf.expand_dims(tf.cast(pos_ss[:1], tf.float32), -1))
    tf.summary.image("pos_se", tf.expand_dims(tf.cast(pos_se[:1], tf.float32), -1))
    tf.summary.image("pos_ee", tf.expand_dims(tf.cast(pos_ee[:1], tf.float32), -1))
    tf.summary.image("pos_es", tf.expand_dims(tf.cast(pos_es[:1], tf.float32), -1))

    # Embedding lookup: batch_size * max_seq_len * max_seq_len * d_model
    pe_ss = tf.nn.embedding_lookup(pos_emb, pos_ss)
    pe_se = tf.nn.embedding_lookup(pos_emb, pos_se)
    pe_ee = tf.nn.embedding_lookup(pos_emb, pos_ee)
    pe_es = tf.nn.embedding_lookup(pos_emb, pos_es)

    # fuse embedding: In original paper only pe_ss, pe_ee are fused
    pe = tf.layers.dense(tf.concat([pe_ss, pe_ee ], axis=-1), units=d_model,
                         activation='relu')
    # pe = tf.layers.dense(tf.concat([pe_ss, pe_se, pe_ee, pe_es], axis=-1), units=d_model,
    #                      activation='relu')
    add_layer_summary('fuse_pe', pe)

    return pe


def relative_attention(key, query, pos_emb, max_seq_len, num_head, per_head_d):
    """
    ATT = QK + QR + uK + vR. Attention is unscaled
    Input:
        key: batch * num_head * max_seq_len * (d_model/num_head)
        query: batch * num_head * max_seq_len * (d_model/num_head)
        pos_emb: batch * num_head * max_seq_len * max_seq_len * d_model
    Relative positional encoding: [-max_seq_len, max_seq_len], in order to distinguish left/right positional
    """
    with tf.variable_scope('tener_relative_attention'):
        u = tf.get_variable(shape=[num_head, per_head_d],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1234),
                            name='content_bias_u')
        v = tf.get_variable(shape=[num_head, per_head_d],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1234),
                            name='positional_bias_v')

        # Content Based Attention Score: QK (query&key semantic) + uK(key bias)
        rw_head_q = query + tf.expand_dims(u, axis=1)
        AC = tf.einsum('bnqd,bnkd->bnqk', rw_head_q, key) # batch * num_head * query_len * key_len
        add_layer_summary('content_base_att', AC)

        # positional Based Attention Score: QR(pos attented query) + vR(pos bias)
        rr_head_q = query + tf.expand_dims(v, axis=1)
        BD = tf.einsum('bnqd,bnqkd->bnqk', rr_head_q, pos_emb) # batch * num_head * query_len * key_len
        add_layer_summary('positinoal_base_att', BD)

        #Att = QK + uK + QR + vR: batch * num_head * query_len * key_len ->(batch*num_head) * query_len * key_len
        weight = AC+BD
        weight = tf.reshape(weight, [-1, max_seq_len, max_seq_len])
        add_layer_summary('att_premask', weight)
    return weight


def relative_multi_head_attention(key, value, query, pos_emb, mask, num_head, dropout_rate, is_training):
    """
    TENER: Adaptive Relative attention
    1. no transformation on Key
    2. Transformer-xl style relative attention, with double side positional encoding
    3. unscaled attention weight
    input:
        key: batch_size * key_len * emb_size
        query: batch_size * query_len * emb_size
        value: batch_size * key_len * emb_size
        mask: batch_size * key_len
    output:
        weighted_val: batch_size * query_len * emb_size
    """
    with tf.variable_scope('multi_head_attention', reuse=tf.AUTO_REUSE):
        d_model = value.shape.as_list()[-1] # emb_size
        max_seq_len = value.shape.as_list()[-2]

        # linear projection with dimension unchanged
        new_pos = tf.layers.dense(pos_emb, units=d_model, activation=None, name='rel_pos_project')
        new_value = tf.layers.dense(value, units=d_model, activation=None, name='pre_value_project')
        new_query = tf.layers.dense(query, units=d_model, activation=None, name='pre_query_project')

        # split d_model by num_head and compute attention in parallel
        # 4dim: batch_size * num_head * key_len * (emb_size/num_head)
        new_key = tf.stack(tf.split(key, num_or_size_splits=num_head, axis=-1), axis=1)
        new_query = tf.stack(tf.split(new_query, num_or_size_splits=num_head, axis=-1), axis=1)
        new_pos = tf.stack(tf.split(new_pos, num_or_size_splits=num_head, axis=-1), axis=1)
        # 3dim: (batch_size * num_head) * key_len * (emb_size/num_head)
        new_value = tf.concat(tf.split(new_value, num_or_size_splits=num_head, axis=-1), axis=0)

        # calculate relative attention
        weight = relative_attention(new_key, new_query, new_pos,
                                    max_seq_len, num_head, int(d_model/num_head))
        weight = normalize_attention(weight, tf.tile(mask, [num_head, 1, 1])) # here mask=seq_len+lex_len
        # weighted value & concat num_head back
        # (batch_size * num_head) * query_len * (emb_size/num_head) -> batch_size * query_len * emb_size
        weighted_val = tf.matmul(weight, new_value)
        weighted_val = tf.concat(tf.split(weighted_val, num_or_size_splits=num_head, axis=0), axis=-1)
        # Linear projection
        weighted_val = tf.layers.dense(weighted_val, units=d_model, activation=None, name='post_linear_project')
        # Do dropout
        weighted_val = tf.layers.dropout(weighted_val, rate=dropout_rate, training=is_training)
        add_layer_summary('raw_multi_head', weighted_val)
        weighted_val = add_and_norm_layer(query, weighted_val)

    return weighted_val
