# -*-coding:utf-8 -*-
from tools.layer import *
from config import TRAIN_PARAMS
from tools.transformer.modules import *
from tools.transformer.encoder import transformer_encoder


def build_graph(features, labels, params, is_training):
    """
    Use pretrain bert input token embedding as input + bilstm + crf
    """
    input_ids = features['token_ids']
    bichar_ids = features['bichar_ids']
    label_ids = features['label_ids']
    seq_len = features['seq_len']

    # generate position embedding lookup table
    pos_seq = np.array(range(params['max_seq_len']))
    pos_encoding = sinusoidal_positional_encoding(params['d_model'], pos_seq)

    with tf.variable_scope('embedding'):
        # Refer to TENER paper we concat char+bichar embedding.
        char_embedding = tf.nn.embedding_lookup(params['embedding'], input_ids)
        bichar_embedding = tf.nn.embedding_lookup(params['bichar_embedding'], bichar_ids)
        embedding = tf.concat([char_embedding, bichar_embedding], axis=-1)
        embedding = embedding_project(embedding, params['d_model'])
        embedding = tf.layers.dropout(embedding, seed=1234, rate=params['embedding_dropout'],
                                      training=is_training)
        add_layer_summary(embedding.name, embedding)

    transformer_output = transformer_encoder(encoder_input=embedding, seq_len=seq_len,
                                             max_seq_len=params['max_seq_len'], encode_layers=params['encode_layers'],
                                             num_head=params['num_head'], dropout_rate=params['dropout_rate'],
                                             ffn_hidden=params['ffn_hidden'], is_training=is_training)

    transformer_output = tf.layers.dropout(transformer_output, seed=1234,
                                           rate=params['dropout_rate'], training=is_training)

    logits = tf.layers.dense(transformer_output, units=params['label_size'], activation=None,
                             use_bias=True, name='logits')
    add_layer_summary(logits.name, logits)

    trans, log_likelihood = crf_layer(logits, label_ids, seq_len, params['label_size'], is_training)
    pred_ids = crf_decode(logits, trans, seq_len, params['idx2tag'], is_training)
    crf_loss = tf.reduce_mean(-log_likelihood)

    return crf_loss, pred_ids

# below params from MSRA
TRANSFORMER_PARAMS = {
    'num_head': 8, # giga embedding size is 50, must be divided by 5
    'd_model': 240,  # giga char& bichar embedding dim are small, project to bigger dim
    'ffn_hidden': 240,
    'encode_layers': 2,
    'batch_size': 16,
    'wramup_ratio': 0.1,
    'epochs': 100
}

TRAIN_PARAMS.update(TRANSFORMER_PARAMS)
TRAIN_PARAMS.update({
    'lr': 0.001,
    'decay_rate': 0.95,  # lr * decay_rate ^ (global_step / train_steps_per_epoch)
    'embedding_dropout': 0.3,
    'dropout_rate': 0.2, # used in transformer sublayer dropout
    'early_stop_ratio': 2 # stop after no improvement after 1.5 epochs
})

