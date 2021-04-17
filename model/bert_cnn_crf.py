# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from config import TRAIN_PARAMS
from tools.utils import add_layer_summary


def build_graph(features, labels, params, is_training):
    """
    pretrain Bert Model output + bilstm + CRF
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']
    input_mask = features['mask']
    segment_ids = features['segment_ids']
    seq_len = features['seq_len']
    embedding = pretrain_bert_embedding(input_ids, input_mask, segment_ids, params['pretrain_dir'],
                                        params['embedding_dropout'], is_training)
    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    cnn_output = cnn_layer(embedding, params['filter_list'], params['kernel_size_list'],
                           params['cnn_activation'], params['cnn_dropout'], is_training)

    logits = tf.layers.dense(cnn_output, units=params['label_size'], activation=None,
                             use_bias=True, name='logits')
    add_layer_summary(logits.name, logits)

    trans, log_likelihood = crf_layer(logits, label_ids, seq_len, params['label_size'], is_training)
    pred_ids = crf_decode(logits, trans, seq_len, params['idx2tag'], is_training)
    crf_loss = tf.reduce_mean(-log_likelihood)

    return crf_loss, pred_ids


CNN_PARAMS = {
    'filter_list': [10],
    'kernel_size_list': [4],
    'padding': 'SAME',
    'cnn_activation': 'relu',
    'cnn_dropout': 0.2
}

TRAIN_PARAMS.update(CNN_PARAMS)
TRAIN_PARAMS.update({
    'diff_lr_times': {'crf': 500,  'logit': 500 , 'lstm': 100,'cnn':100}
})
