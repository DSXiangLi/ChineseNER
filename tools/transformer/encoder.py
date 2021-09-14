# -*-coding:utf-8 -*-
from tools.transformer.modules import *
from tools.transformer.tener import relative_multi_head_attention


def transformer_encoder(encoder_input, seq_len, max_seq_len, encode_layers,
                        num_head, dropout_rate, ffn_hidden, is_training):
    with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
        self_mask = seq_mask_gen(seq_len, max_seq_len)
        for i in range(encode_layers):
            with tf.variable_scope('self_attention_layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                encoder_input = multi_head_attention(key=encoder_input, query=encoder_input, value=encoder_input,
                                                     mask=self_mask, num_head=num_head,
                                                     dropout_rate=dropout_rate, is_training=is_training)
                add_layer_summary('output', encoder_input)
                encoder_input = ffn(encoder_input, ffn_hidden, dropout_rate, is_training)
                add_layer_summary('ffn', encoder_input)
    return encoder_input


def tener_encoder(encoder_input, seq_len, max_seq_len, encode_layers,
                 num_head, dropout_rate, ffn_hidden, is_training):
    with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
        self_mask = seq_mask_gen(seq_len, max_seq_len)
        for i in range(encode_layers):
            with tf.variable_scope('self_attention_layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                encoder_input = relative_multi_head_attention(key=encoder_input, query=encoder_input, value=encoder_input,
                                                     mask=self_mask, num_head=num_head,
                                                     dropout_rate=dropout_rate, is_training=is_training)
                add_layer_summary('output', encoder_input)
                encoder_input = ffn(encoder_input, ffn_hidden, dropout_rate, is_training)
                add_layer_summary('ffn', encoder_input)
    return encoder_input