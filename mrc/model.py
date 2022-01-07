# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary


def build_graph(features, labels,  params, is_training):
    """
    pretrain Bert model output + CRF Layer
    """
    input_ids = features['input_ids']
    segment_ids = features['segment_ids']
    query_len = features['query_len']
    text_len = features['text_len']

    max_seq_len = tf.reduce_max(query_len+text_len)
    input_mask = tf.sequence_mask(query_len + text_len, maxlen=max_seq_len)

    embedding = pretrain_bert_embedding(input_ids, input_mask, segment_ids, params['pretrain_dir'],
                                        params['embedding_dropout'], is_training)
    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    # BIO prediction: batch * max_seq * label_size
    logits = tf.layers.dense(embedding, units=params['label_size'], activation=None, use_bias=True, name='logits')
    probs = tf.nn.softmax(logits)

    with tf.variable_scope('mask'):
        # Generate Mask for text
        query_mask = tf.cast(tf.sequence_mask(query_len, maxlen=max_seq_len), tf.float32)
        text_mask = tf.cast(input_mask, tf.float32) - query_mask

        add_layer_summary('query_mask', query_mask)
        add_layer_summary('text_mask', text_mask)

    if labels is None:
        return None, probs, text_mask

    loss_func = params['loss_func']
    loss = loss_func(logits, labels['label_ids'], text_mask)

    return loss, probs, text_mask


def cross_entropy_loss_mask(logits, label_ids, mask):
    with tf.variable_scope('ce_loss'):
        label_size = logits.get_shape().as_list()[-1]
        # flatten predict and label
        logits = tf.reshape(logits, [-1, label_size])
        label_ids = tf.reshape(label_ids, [-1])
        # calc cross-entropy loss
        mask = tf.cast(mask, tf.float32)
        loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
        loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))
        loss = tf.reduce_sum(loss_mat)/tf.reduce_sum(mask + 1e-10) # span mask可能出现为0的情况
    return loss


def calc_metrics(label_ids, probs, weight, prefix):
    # cast all to N-class dim
    num_labels = probs.get_shape().as_list()[-1]
    pred_ids = tf.argmax(probs, axis=-1)
    pred_ids = tf.one_hot(pred_ids, depth=num_labels)
    label_ids = tf.one_hot(label_ids, depth=num_labels)

    precision, precision_op = tf.metrics.precision(
            labels=label_ids, predictions=pred_ids, weights=weight)
    recall, recall_op = tf.metrics.recall(
            labels=label_ids, predictions=pred_ids, weights=weight)
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics = {

        'metric/{}_auc'.format(prefix): tf.metrics.auc(labels=label_ids, predictions=probs, curve='ROC',
                                                       weights=weight,
                                                       summation_method='careful_interpolation'),
        'metric/{}_ap'.format(prefix): tf.metrics.auc(labels=label_ids, predictions=probs, curve='PR',
                                                     weights=weight,
                                                     summation_method='careful_interpolation'),
        'metric/{}_accuracy'.format(prefix): tf.metrics.accuracy(
            labels=label_ids, predictions=pred_ids, weights=weight),
        'metric/{}_precision'.format(prefix): (precision, precision_op),
        'metric/{}_recall'.format(prefix): (recall, recall_op),
        'metric/{}_f1'.format(prefix): (f1, tf.identity(f1))
    }
    return metrics


def build_model_fn():
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        loss, probs, text_mask = build_graph(features, labels, params,  is_training)

        if is_training:
            train_op = bert_train_op(loss, params['lr'], params['num_train_steps'],
                                     params['warmup_ratio'], params['diff_lr_times'], True)
            spec = tf.estimator.EstimatorSpec(mode, loss=loss,
                                              train_op=train_op,
                                              training_hooks=[get_log_hook(loss, params['log_steps'])])
        elif mode == tf.estimator.ModeKeys.EVAL:
            metric_op = {}
            metric_op.update(calc_metrics(labels['label_ids'], probs, text_mask, 'bio'))
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_op)
        else:
            spec = tf.estimator.EstimatorSpec(mode, predictions={'probs': probs,
                                                                 'text_mask': text_mask})
        return spec

    return model_fn


if __name__ == '__main__':
    labels = {'start_ids': tf.constant([[0, 1, 0, 0, 0,0], [0,0,1,0,0,0]]),
              'end_ids': tf.constant([[0, 0, 0, 1, 0,0], [0,0,1,0,0,0]])}
    start_probs = tf.constant([[1,0,0,0,0,0], [0,0,1,0,0,0]], tf.float32)
    end_probs = tf.constant([[0,0,0,1,0,0,],[0,0,1,1,0,0]], tf.float32)
    query_len = tf.constant([1,2])
    text_len = tf.constant([4,3])
    max_seq_len = 6