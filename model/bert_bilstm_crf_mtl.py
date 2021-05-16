# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS


def build_graph(features, labels, params, is_training):
    """
    Multi-task learning. task can be CWS + NER， or different NER dataset
    asymmetry=False, all task share bert embedding, and has its own bilstm+crf tower
    asymmetry=True, task2 is the main task, 2 task share bert embedding, task2 use task 1 hidden state also
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']
    input_mask = features['mask']
    segment_ids = features['segment_ids']
    seq_len = features['seq_len']
    task_ids = features['task_ids']

    embedding = pretrain_bert_embedding(input_ids, input_mask, segment_ids, params['pretrain_dir'],
                                        params['embedding_dropout'], is_training)

    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    mask1 = tf.equal(task_ids, 0)
    mask2 = tf.equal(task_ids, 1)
    batch_size = tf.shape(task_ids)[0]
    with tf.variable_scope(params['task_list'][0], reuse=tf.AUTO_REUSE):
        task_params = params[params['task_list'][0]]
        lstm_output1 = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                             params['hidden_units_list'], params['keep_prob_list'],
                             params['cell_size'], params['dtype'], is_training)

        logits = tf.layers.dense(lstm_output1, units=task_params['label_size'], activation=None,
                                 use_bias=True, name='logits')
        add_layer_summary(logits.name, logits)

        trans1, loglikelihood1 = crf_layer(logits, label_ids, seq_len, task_params['label_size'], is_training)
        pred_ids1 = crf_decode(logits, trans1, seq_len, task_params['idx2tag'], is_training, mask1)

        loss1 = tf.reduce_sum(tf.boolean_mask(-loglikelihood1, mask1, axis=0)) * params['task_weight'][0]
        tf.summary.scalar('loss', loss1)

    with tf.variable_scope(params['task_list'][1], reuse=tf.AUTO_REUSE):
        task_params = params[params['task_list'][1]]
        lstm_output2 = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                             params['hidden_units_list'], params['keep_prob_list'],
                             params['cell_size'], params['dtype'], is_training)

        if params['asymmetry']:
            # if asymmetry, task2 is the main task using task1 information
            lstm_output2 = tf.concat([lstm_output1, lstm_output2], axis=-1)
        logits = tf.layers.dense(lstm_output2, units=task_params['label_size'], activation=None,
                                 use_bias=True, name='logits')
        add_layer_summary(logits.name, logits)

        trans2, loglikelihood2 = crf_layer(logits, label_ids, seq_len, task_params['label_size'], is_training)
        pred_ids2 = crf_decode(logits, trans2, seq_len, task_params['idx2tag'], is_training, mask2)

        loss2 = tf.reduce_sum(tf.boolean_mask(-loglikelihood2, mask2, axis=0)) * params['task_weight'][1]
        tf.summary.scalar('loss', loss2)

    loss = (loss1+loss2)/tf.cast(batch_size, dtype=params['dtype'])
    pred_ids = tf.where(tf.equal(task_ids, 0), pred_ids1, pred_ids2) # for infernce all pred_ids will be for 1 task
    return loss, pred_ids, task_ids


RNN_PARAMS = {
    'cell_type': 'lstm',
    'cell_size': 1,
    'hidden_units_list': [128],
    'keep_prob_list': [0.8],
    'rnn_activation': 'relu',
    'batch_size': 32
}

TRAIN_PARAMS.update(RNN_PARAMS)
TRAIN_PARAMS.update({
    'diff_lr_times': {'crf': 500,  'logit': 500 , 'lstm': 100},
    'task_weight': [1, 1], # equal weight for CWS+NER/NER+NER task，
    'asymmetry': True # If asymmetry, task2 is the main task, using task1 hidden information
})
