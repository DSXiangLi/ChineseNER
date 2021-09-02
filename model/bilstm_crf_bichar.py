# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS


def build_graph(features, labels, params, is_training):
    """
    Giga pretrain character embedding+  bilstm + CRF + softword word enhance
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']

    seq_len = features['seq_len']
    bichar_ids = features['bichar_ids']

    with tf.variable_scope('embedding'):
        embedding = tf.nn.embedding_lookup(params['embedding'], input_ids)
        embedding = tf.layers.dropout(embedding, rate=params['embedding_dropout'],
                                      seed=1234, training=is_training)
        add_layer_summary(embedding.name, embedding) # emb_dim=50

    with tf.variable_scope('word_enhance'):
        wh_embedding = tf.nn.embedding_lookup(params['bichar_embedding'], bichar_ids) # max_seq_len * emb_dim
        wh_embedding = tf.layers.dropout(wh_embedding, rate=params['embedding_dropout'],
                                         seed=1234, training=is_training)
        embedding = tf.concat([embedding, wh_embedding], axis=-1)
        add_layer_summary(wh_embedding.name, wh_embedding)

    lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                         params['hidden_units_list'], params['keep_prob_list'],
                         params['cell_size'], seq_len, params['dtype'], is_training)

    lstm_output = tf.layers.dropout(lstm_output, seed=1234, rate=params['embedding_dropout'],
                                      training=is_training)

    add_layer_summary(lstm_output.name, lstm_output)

    logits = tf.layers.dense(lstm_output, units=params['label_size'], activation=None,
                             use_bias=True, name='logits')
    add_layer_summary(logits.name, logits)

    trans, log_likelihood = crf_layer(logits, label_ids, seq_len, params['label_size'], is_training)
    pred_ids = crf_decode(logits, trans, seq_len, params['idx2tag'], is_training)
    crf_loss = tf.reduce_mean(-log_likelihood)

    return crf_loss, pred_ids


RNN_PARAMS = {
    'cell_type': 'lstm',
    'cell_size': 1,
    'hidden_units_list': [128],
    'keep_prob_list': [1],
    'rnn_activation': 'tanh'
}

TRAIN_PARAMS.update(RNN_PARAMS)
TRAIN_PARAMS.update({
    'lr': 0.005,
    'decay_rate': 0.95,  # lr * decay_rate ^ (global_step / train_steps_per_epoch)
    'embedding_dropout': 0.2,
    'early_stop_ratio': 2 # stop after no improvement after 1.5 epochs
})
