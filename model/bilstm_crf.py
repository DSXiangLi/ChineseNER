# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS


def build_graph(features, labels, params, is_training):
    """
    Use pretrain bert input token embedding as input + bilstm + crf
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']
    seq_len = features['seq_len']

    # use bert input embedding
    # embedding = bert_token_embedding(input_ids, params['pretrain_dir'],
    #                                  params['embedding_dropout'], is_training)
    #
    # load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    # use pretrain character embedding
    with tf.variable_scope('embedding'):
        embedding = tf.nn.embedding_lookup(params['embedding'], input_ids)
        embedding = tf.layers.dropout(embedding, seed=1234, rate=params['embedding_dropout'],
                                      training=is_training)
        add_layer_summary(embedding.name, embedding)

    lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                         params['hidden_units_list'], params['keep_prob_list'],
                         params['cell_size'], seq_len, params['dtype'], is_training)

    lstm_output = tf.layers.dropout(lstm_output, seed=1234, rate=params['embedding_dropout'],
                                      training=is_training)

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
    'embedding_dropout': 0.3,
    'early_stop_ratio': 2 # stop after no improvement after 1.5 epochs
})
