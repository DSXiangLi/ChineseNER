# -*-coding:utf-8 -*-
"""
    Modified positional encoding from TENER
    Github Reference:
"""
import tensorflow as tf
import numpy as np
from tools.utils import add_layer_summary
from tools.transformer.modules import add_and_norm_layer, normalize_attention, sinusoidal_positional_encoding


def relative_attention(key, query, num_head, d_model):
    """
    ATT = QK + QR + uK + vR. Attention is unscaled
    Input:
        key: batch * num_head * max_seq_len * (d_model/num_head)
        query: batch * num_head * max_seq_len * (d_model/num_head)
    Relative positional encoding: [-max_seq_len, max_seq_len], in order to distinguish left/right positional
    """

    with tf.variable_scope('tener_relative_attention'):
        max_seq_len = query.shape.as_list()[-2]
        pos_seq = np.array(range(-max_seq_len, max_seq_len))
        pos_emb = sinusoidal_positional_encoding(d_model, pos_seq) # (2*max_seq_len) * d_model

        u = tf.get_variable(shape=[num_head, d_model],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1234),
                            name='content_bias_u')
        v = tf.get_variable(shape=[num_head, d_model],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1234),
                            name='positional_bias_v')

        # Content Based Attention Score: QK (query&key semantic) + uK(key bias)
        rw_head_q = query + tf.expand_dims(u, axis=1)
        AC = tf.einsum('bnqd,bnkd->bnqk', rw_head_q, key) # batch * num_head * query_len * key_len
        add_layer_summary('content_base_att', AC)

        # positional Based Attention Score: QR(pos attented query) + vR(pos bias)
        rr_head_q = query + tf.expand_dims(v, axis=1)
        BD = tf.einsum('bnqd,ld->bnql', rr_head_q, pos_emb) # batch * num_head * query_len * (2 * max_seq_len)
        BD = shift(BD)
        add_layer_summary('positinoal_base_att', BD)

        #Att = QK + uK + QR + vR: batch * num_head * query_len * key_len ->(batch*num_head) * query_len * key_len
        weight = AC+BD
        weight = tf.reshape(weight, [-1, max_seq_len, max_seq_len])
        add_layer_summary('att_premask', weight)
    return weight


def shift(BD):
    """
    Input:
        BD: batch * num_head * query_len * (2* max_seq_len)
        E.g max_seq_len =3
        -3 -2 -1 [0 1 2]
        -3 -2 [-1 0 1] 2
        -3 [-2 -1 0] 1 2
    Output:
        batch * num_head * query_len * key_len [for self-encoding max_seq_len=query_len=key_len]
        通过加入zero-pad以及reshape变化，截取以上[]的部分，哇塞超级巧妙～.～
        0   1  2
        -1  0  1
        -2 -1  0
    """
    _, num_head, query_len, pos_len = BD.shape.as_list()
    # 右边加入一列zero-pad: b * n * l* 2l-> b * n * l * (2l+1)
    BD = tf.concat([BD, tf.zeros_like(BD[:,:,:,:1], dtype=tf.float32)], axis=-1)
    #这一步reshape是精髓! b * n * l * (2l+1)->b * n * (2l) * l因为zero产生错位
    BD = tf.reshape(BD, [-1, num_head, (pos_len+1), query_len])[:, :, :-1]
    #reshape回原始size: b * n * l * 2l
    BD = tf.reshape(BD, (-1, num_head, query_len, pos_len))
    BD = BD[:, :, :, query_len:]
    return BD


def relative_multi_head_attention(key, value, query, mask, num_head, dropout_rate, is_training):
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
        # linear projection with dimension unchaangned
        new_key = key # relative attention keeps key unchanged
        new_value = tf.layers.dense(value, units=d_model, activation=None, name='pre_value_project')
        new_query = tf.layers.dense(query, units=d_model, activation=None, name='pre_query_project')

        # split d_model by num_head and compute attention in parallel
        # 4dim: batch_size * num_head * key_len * (emb_size/num_head)
        new_key = tf.stack(tf.split(new_key, num_or_size_splits=num_head, axis=-1), axis=1)
        new_query = tf.stack(tf.split(new_query, num_or_size_splits=num_head, axis=-1), axis=1)
        # 3dim: (batch_size * num_head) * key_len * (emb_size/num_head)
        new_value = tf.concat(tf.split(new_value, num_or_size_splits=num_head, axis=-1), axis=0)

        # calculate relative attention
        weight = relative_attention(new_key, new_query, num_head, int(d_model/num_head))
        weight = normalize_attention(weight, tf.tile(mask, [num_head, 1, 1]))
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


if __name__ =='__main__':
    BD = tf.constant([[[[-3, -2, -1, 0, 1, 2],
                        [-3, -2, -1, 0, 1, 2],
                        [-3, -2, -1, 0, 1, 2]]]])
    a = shift(BD)
    sess = tf.Session()
    sess.run(a)