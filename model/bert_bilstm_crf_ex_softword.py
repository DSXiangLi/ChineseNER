# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS


def build_graph(features, labels, params, is_training):
    """
    pretrain Bert Model output + bilstm + CRF
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']
    input_mask = features['mask']
    segment_ids = features['segment_ids']
    seq_len = features['seq_len']
    ex_softword_ids = features['ex_softword_ids'] # bathc * ma_seq * word_enhance_dim

    if params['use_bert']:
        # use bert model
        embedding = pretrain_bert_embedding(input_ids, input_mask, segment_ids, params['pretrain_dir'],
                                            params['embedding_dropout'], is_training)
    else:
        # not use bert_model, need to rename model name
        embedding = bert_token_embedding(input_ids, params['pretrain_dir'],
                                         params['embedding_dropout'], is_training)
    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    with tf.variable_scope('word_enhance'):
        emb_dim = embedding.shape.as_list()[-1]
        softword_embedding = tf.get_variable(
            shape=[params['word_enhance_dim'], emb_dim],
            initializer=tf.truncated_normal_initializer(), name='ex_softword_embedding')
        wh_embedding = tf.matmul(ex_softword_ids, softword_embedding) #max_seq_len * emb_dim
        #embedding = tf.concat([wh_embedding, embedding], axis=-1) # concat word enhance with token embedding
        embedding += wh_embedding # adding seems to be better than concat
        add_layer_summary(softword_embedding.name, softword_embedding)
        add_layer_summary(embedding.name, embedding)

    lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                         params['hidden_units_list'], params['keep_prob_list'],
                         params['cell_size'], params['dtype'], is_training)

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
    'keep_prob_list': [0.8],
    'rnn_activation': 'relu'
}

TRAIN_PARAMS.update(RNN_PARAMS)
TRAIN_PARAMS.update({
    'diff_lr_times': {'crf': 500,  'logit': 500, 'lstm': 100, 'word_enhance': 100},
    'use_bert': False # for word enhance model we compare with/without bert
})
