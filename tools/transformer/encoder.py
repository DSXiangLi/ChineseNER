# -*-coding:utf-8 -*-
from tools.transformer.modules import *


def transformer_encoder(encoder_input, seq_len, params, is_training):
    with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
        self_mask = seq_mask_gen(seq_len, params['max_seq_len'], params)
        for i in range(params['encode_attention_layers']):
            with tf.variable_scope('self_attention_layer_{}'.format(i), reuse=tf.AUTO_REUSE):
                encoder_input = multi_head_attention(key=encoder_input, query=encoder_input, value=encoder_input,
                                                     mask=self_mask, params=params, is_training=is_training)
                add_layer_summary('output', encoder_input)
                encoder_input = ffn(encoder_input, params, is_training)
                add_layer_summary('ffn', encoder_input)
    return encoder_input

