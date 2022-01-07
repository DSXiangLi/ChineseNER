# -*-coding:utf-8 -*-
import inspect
import tensorflow as tf
from tools.train_utils import AddonParser

LossHP = {}
LossFunc = {}


def extend_addon_wrapper(func):
    """
    Loss Function 推荐参数直接从function default argument里面读取
    """
    global LossHP, LossFunc
    hp = []
    sg = inspect.signature(func)
    for k, v in sg.parameters.items():
        hp.append(AddonParser.hp(k, v.default))
    LossHP[func.__name__] = AddonParser(hp)
    LossFunc[func.__name__] = func


def cross_entropy_loss(logits, label_ids, seq_len, label_size, max_seq_len, dtype):
    with tf.variable_scope('ce_loss'):
        # flatten predict and label
        logits = tf.reshape(logits, [-1, label_size])
        label_ids = tf.reshape(label_ids, [-1])
        # generate padding mask
        mask = tf.sequence_mask(lengths=seq_len, maxlen=max_seq_len, dtype=dtype)
        # calc cross-entropy loss
        loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
        loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))
        loss = tf.reduce_sum(loss_mat)/tf.reduce_sum(mask)
    return loss


@extend_addon_wrapper
def ce():
    """
    Cross-Entropy Loss
    """
    def helper(logits, label_ids, mask):
        with tf.variable_scope('ce_loss'):
            label_size = logits.get_shape().as_list()[-1]
            # flatten predict and label
            logits = tf.reshape(logits, [-1, label_size])
            label_ids = tf.reshape(label_ids, [-1])
            # calc cross-entropy loss
            mask = tf.cast(mask, tf.float32)
            loss_mat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
            loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))
            loss = tf.reduce_sum(loss_mat) / tf.reduce_sum(mask + 1e-10)  # span mask可能出现为0的情况
        return loss
    return helper


@extend_addon_wrapper
def focal(gamma=2):
    """
    2017-Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    """
    def helper(logits, label_ids, mask):
        """
        输入和tf.nn.sparse_softmax_cross_entropy_with_logits保持一致
        """
        with tf.variable_scope('focal_loss'):
            label_size = logits.get_shape().as_list()[-1]
            # flatten predict and label
            logits = tf.reshape(logits, [-1, label_size])
            probs = tf.nn.softmax(logits, axis=-1)

            # transform to one-hot
            label_ids = tf.reshape(label_ids, [-1])
            label_ids = tf.one_hot(label_ids, depth=label_size)

            #focal loss
            loss_mat = tf.reduce_sum(label_ids * tf.pow((1 - probs), gamma) * tf.log(probs), axis=-1)
            mask = tf.cast(mask, tf.float32)
            loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))
            loss = -tf.reduce_sum(loss_mat) / tf.reduce_sum(mask + 1e-10)
        return loss
    return helper


@extend_addon_wrapper
def gce(q=0.7):
    """
    2018-Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
    https://arxiv.org/abs/1805.07836
    q: normalized factor
    """
    def helper(logits, label_ids, mask):
        with tf.variable_scope('gcec_loss'):
            label_size = logits.get_shape().as_list()[-1]
            # flatten predict and label
            logits = tf.reshape(logits, [-1, label_size])
            probs = tf.nn.softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_ids = tf.one_hot(label_ids, depth=label_size)

            # calc generalized cross entropy
            mask = tf.cast(mask, tf.float32)
            loss_mat = (1-tf.pow(tf.reduce_sum(label_ids * probs, axis=-1), q))/q
            loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))
            loss = tf.reduce_sum(loss_mat) / tf.reduce_sum(mask + 1e-10)  # span mask可能出现为0的情况
        return loss
    return helper


@extend_addon_wrapper
def sce(alpha=0.1, beta=1):
    """
    2018-Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
    https://arxiv.org/abs/1805.07836
    Input:
    beta: coefficient for rce
    alpha: coefficient for ce
    """
    def helper(logits, label_ids, mask):
        with tf.variable_scope('gcec_loss'):
            label_size = logits.get_shape().as_list()[-1]
            # flatten predict and label
            logits = tf.reshape(logits, [-1, label_size])
            probs = tf.nn.softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_ids = tf.one_hot(label_ids, depth=label_size)
            # calc generalized cross entropy
            mask = tf.cast(mask, tf.float32)
            loss_mat = (1-tf.pow(tf.reduce_sum(label_ids * probs,axis =-1), q))/q
            loss_mat = tf.multiply(loss_mat, tf.reshape(mask, [-1]))
            loss = tf.reduce_sum(loss_mat) / tf.reduce_sum(mask + 1e-10)  # span mask可能出现为0的情况
        return loss
    return helper


