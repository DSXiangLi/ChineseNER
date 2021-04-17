# -*-coding:utf-8 -*-
import tensorflow as tf


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


def dice_loss(logits, label_ids, seq_len, idx2tag, max_seq_len, alpha, gamma):
    """
    logits: batch_size * max_len * num_tag
    label_ids : batch_size * max_len
    alpha: push down easy sample
    gamma: smoothing
    """

    def calc_dsc(input_prob, input_label, mask, alpha, gamma):
        prob_with_factor = (1 - input_prob) ** alpha * input_prob

        nominator = 2 * prob_with_factor * input_label + gamma
        denominator = prob_with_factor + input_label + gamma

        dsc = nominator / denominator * mask  # batch_size * max_seq

        return 1 - tf.reduce_sum(dsc) / tf.reduce_sum(mask)

    with tf.variable_scope('dsc_layer'):
        mask = tf.sequence_mask(seq_len, maxlen=max_seq_len, dtype=tf.float32)
        probs = tf.nn.softmax(logits, axis=-1) # normalize logits to probs
        loss = 0
        for i,j in idx2tag.items():
            if j not in ['[PAD]', '[CLS]', '[SEP]','O']:
                input_prob = tf.gather(probs, tf.cast(i, tf.int32), axis=-1)  # batch * max_seq
                input_label = tf.cast(tf.equal(label_ids, i), tf.float32) # batch * max_seq
                loss += calc_dsc(input_prob, input_label, mask, alpha, gamma)
    return loss
