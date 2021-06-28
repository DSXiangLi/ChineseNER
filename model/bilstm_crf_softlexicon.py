# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS


def build_graph(features, labels, params, is_training):
    """
    Giga pretrain character embedding+  bilstm + CRF + softlexicon word enhance
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']

    seq_len = features['seq_len']
    softlexicon_ids = features['softlexicon_ids'] # batch_size * max_seq_len *(len(soft2idx) * max_lexicon_len)
    softlexicon_weights = features['softlexicon_weights'] # same size as softlexicon_ids

    with tf.variable_scope('embedding'):
        embedding = tf.nn.embedding_lookup(params['embedding'], input_ids)
        embedding = tf.layers.dropout(embedding, rate=params['embedding_dropout'],
                                      seed=1234, training=is_training)
        add_layer_summary(embedding.name, embedding)

    with tf.variable_scope('word_enhance'):
        # Init word embedding with pretrain word2vec model
        softword_embedding = tf.get_variable(initializer=params['word_embedding'],
                                             dtype=params['dtype'],
                                             name='softlexicon_embedding')
        word_embedding_dim = softword_embedding.shape.as_list()[-1]
        wh_embedding = tf.nn.embedding_lookup(softword_embedding, softlexicon_ids) # max_seq_len * 50(MaxLexicon * len(SoftIdx)) * emb_dim
        wh_embedding = tf.multiply(wh_embedding, tf.expand_dims(softlexicon_weights, axis=-1))
        # Method1: weighted average lexicons in each B/M/E/S/None and concatenate -> 5 * emb_dim
        wh_embedding = tf.reshape(wh_embedding, [-1, params['max_seq_len'], params['word_enhance_dim'] ,
                                                 params['max_lexicon_len'], word_embedding_dim])
        wh_embedding = tf.reduce_mean(wh_embedding, axis=-2)
        wh_embedding = tf.reshape(wh_embedding, [-1, params['max_seq_len'],
                                                 int(params['word_enhance_dim'] * word_embedding_dim)])

        wh_embedding = tf.layers.dropout(wh_embedding, rate=params['embedding_dropout'],
                                         seed=1234, training=is_training)
        embedding = tf.concat([wh_embedding, embedding], axis=-1)
        add_layer_summary('wh_embedding', wh_embedding)
        add_layer_summary(embedding.name, embedding)

    lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                         params['hidden_units_list'], params['keep_prob_list'],
                         params['cell_size'], seq_len, params['dtype'], is_training)

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
    'rnn_activation': 'tanh',
}


TRAIN_PARAMS.update(RNN_PARAMS)
TRAIN_PARAMS.update({
    'lr': 0.005,
    'decay_rate': 0.95,  # lr * decay_rate ^ (global_step / train_steps_per_epoch)
    'embedding_dropout': 0.2,
    'early_stop_ratio': 2 # stop after no improvement after 1.5 epochs
})
