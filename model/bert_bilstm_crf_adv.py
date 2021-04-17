# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS

flip_gradient = FlipGradientBuilder()

def build_graph(features, labels, params, is_training):
    """
    Adversarial Training. task can be CWS + NER， or different NER dataset
    all task share bert embedding, and has its own bilstm+crf layer
    Equal weight for all task, with lambda weight for discriminator
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

    with tf.variable_scope('task_discriminator', reuse=tf.AUTO_REUSE):
        share_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                             params['hidden_units_list'], params['keep_prob_list'],
                             params['cell_size'], params['dtype'], is_training) # batch * max_seq * (2*hidden)
        share_max_pool = tf.reduce_max(share_output, axis=1, name='share_max_pool') # batch * (2* hidden) extract most significant feature
        # reverse gradient of max_output to only update the unit use to distinguish task
        share_max_pool = flip_gradient(share_max_pool, params['shrink_gradient_reverse'])
        share_max_pool = tf.layers.dropout(share_max_pool, rate=params['share_dropout'],
                                           seed=1234, training=is_training)
        add_layer_summary(share_max_pool.name, share_max_pool)

        logits = tf.layers.dense(share_max_pool, units=len(params['task_list']), activation=None,
                                 use_bias=True, name='logits')# batch * num_task
        add_layer_summary(logits.name, logits)

        adv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=features['task_ids'], logits=logits)
        adv_loss = tf.reduce_mean(adv_loss, name='loss')
        tf.summary.scalar('loss', adv_loss)

    with tf.variable_scope('task1_{}'.format(params['task_list'][0]), reuse=tf.AUTO_REUSE):
        task_params = params[params['task_list'][0]]
        lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                             params['hidden_units_list'], params['keep_prob_list'],
                             params['cell_size'], params['dtype'], is_training)
        lstm_output = tf.concat([share_output, lstm_output], axis=-1) # bath * (4* hidden)

        logits = tf.layers.dense(lstm_output, units=task_params['label_size'], activation=None,
                                 use_bias=True, name='logits')
        add_layer_summary(logits.name, logits)

        trans1, loglikelihood1 = crf_layer(logits, label_ids, seq_len, task_params['label_size'], is_training)
        pred_ids1 = crf_decode(logits, trans1, seq_len, task_params['idx2tag'], is_training, mask1)

        loss1 = tf.reduce_sum(tf.boolean_mask(-loglikelihood1, mask1, axis=0)) * params['task_weight'][0]
        tf.summary.scalar('loss', loss1)

    with tf.variable_scope('task2_{}'.format(params['task_list'][1]), reuse=tf.AUTO_REUSE):
        task_params = params[params['task_list'][1]]
        lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                             params['hidden_units_list'], params['keep_prob_list'],
                             params['cell_size'], params['dtype'], is_training)
        lstm_output = tf.concat([share_output, lstm_output], axis=-1) # bath * (4* hidden)

        logits = tf.layers.dense(lstm_output, units=task_params['label_size'], activation=None,
                                 use_bias=True, name='logits')
        add_layer_summary(logits.name, logits)

        trans2, loglikelihood2 = crf_layer(logits, label_ids, seq_len, task_params['label_size'], is_training)
        pred_ids2 = crf_decode(logits, trans2, seq_len, task_params['idx2tag'], is_training, mask2)

        loss2 = tf.reduce_sum(tf.boolean_mask(-loglikelihood2, mask2, axis=0)) * params['task_weight'][1]
        tf.summary.scalar('loss', loss2)

    loss = (loss1+loss2)/tf.cast(batch_size, dtype=params['dtype']) + adv_loss * params['lambda']
    pred_ids = tf.where(tf.equal(task_ids, 0), pred_ids1, pred_ids2)

    return loss, pred_ids, task_ids


RNN_PARAMS = {
    'cell_type': 'lstm',
    'cell_size': 1,
    'hidden_units_list': [100],
    'keep_prob_list': [0.8],
    'rnn_activation': 'relu'
}

TRAIN_PARAMS.update(RNN_PARAMS)
TRAIN_PARAMS.update({
    'diff_lr_times': {'crf': 500,  'logit': 100 , 'lstm': 100},
    'lambda': 0.5, # weight of task discriminator, can be tuned
    'task_weight': [1,1],  # weight for 2 task
    'shrink_gradient_reverse': 0.001, # CWS+NER task0.01, NER+NER task 0.001, can be tuned.
    'share_dropout': 0.2,
    'batch_size': 32
})
