# -*-coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from bert_base.bert import modeling
from tools.train_utils import map2sequence
from tools.utils import add_layer_summary


def build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size):
    if cell_type.lower() == 'rnn':
        cell_class = tf.nn.rnn_cell.RNNCell
    elif cell_type.lower() == 'gru':
        cell_class = tf.nn.rnn_cell.GRUCell
    elif cell_type.lower() == 'lstm':
        cell_class = tf.nn.rnn_cell.LSTMCell
    else:
        raise Exception('Only rnn, gru, lstm are supported as cell_type')

    return tf.nn.rnn_cell.MultiRNNCell(
        cells = [ tf.nn.rnn_cell.DropoutWrapper(cell = cell_class(num_units = hidden_units_list[i], activation=activation),
                                                output_keep_prob=keep_prob_list[i],
                                                state_keep_prob=keep_prob_list[i]) for i in range(cell_size) ]
    )


def bilstm(embedding, cell_type, activation, hidden_units_list, keep_prob_list, cell_size, seq_len, dtype, is_training):
    with tf.variable_scope('bilstm_layer'):
        if not is_training:
            keep_prob_list = len(keep_prob_list) * [1.0]
        fw = build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size)
        bw = build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size)

        # tuple of 2 : batch_size * max_seq_len * hidden_size
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, embedding, seq_len, dtype=dtype)

        # concat forward and backward along embedding axis
        outputs = tf.concat(outputs, axis=-1)  # batch_size * max_seq_len * (hidden_size * 2)
        add_layer_summary('bilstm_concat', outputs)

    return outputs


def cnn_layer(embedding, filter_list, kernel_size_list, activation, drop_out, is_training):
    outputs = []
    for i in range(len(filter_list)):
        # batch * max_seq_len * filters
        output = tf.layers.conv1d(
            inputs=embedding,
            filters= filter_list[i],
            kernel_size=kernel_size_list[i],
            padding='SAME', # for seq label task, max_seq_len can't change
            activation=activation,
            name='cnn_kernel{}'.format(kernel_size_list[i])
        )
        add_layer_summary(output.name, output)
        output = tf.layers.dropout(output, rate=drop_out, seed=1234, training=is_training)
        outputs.append(output)
    output = tf.concat(outputs, axis=-1) # batch_size * max_seq_len * sum(filter_list)
    return output


def pretrain_bert_embedding(input_ids, input_mask, segment_ids, pretrain_dir, drop_out, is_training):
    # use bert pretrain output from last stack
    # !!! Don't add additional variable_scope, will case bert checkpoint init failed
    bert_config = modeling.BertConfig.from_json_file(os.path.join(pretrain_dir,'bert_config.json'))

    bert_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        scope='bert'
    )
    embedding = bert_model.get_sequence_output()
    embedding = tf.layers.dropout(embedding, rate=drop_out, seed=1234, training=is_training)
    add_layer_summary(embedding.name, embedding)

    return embedding


def bert_token_embedding(input_ids, pretrain_dir, drop_out, is_training):
    bert_config = modeling.BertConfig.from_json_file(os.path.join(pretrain_dir, 'bert_config.json'))
    if not is_training:
        dropout_prob = 0
    else:
        dropout_prob = bert_config.hidden_dropout_prob

    embedding, _ = modeling.embedding_lookup(
        input_ids=input_ids,
        vocab_size=bert_config.vocab_size,
        embedding_size=bert_config.hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=False)

    embedding = modeling.embedding_postprocessor(
        input_tensor=embedding,
        use_token_type=False,
        use_position_embeddings=True,
        position_embedding_name="position_embeddings",
        initializer_range=bert_config.initializer_range,
        dropout_prob=dropout_prob)

    add_layer_summary(embedding.name, embedding)
    embedding = tf.layers.dropout(embedding, rate=drop_out, seed=1234, training=is_training)
    return embedding


def crf_layer(logits, label_ids, seq_len, label_size, is_training):
    with tf.variable_scope('crf_layer'):
        trans = tf.get_variable(
            "transitions",
            shape=[label_size, label_size],
            initializer=tf.contrib.layers.xavier_initializer())

        if label_ids is None:
            return trans, None
        # real length of sequence to compute log-likelihood, include CLS and SEP
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=label_ids,
            transition_params=trans,
            sequence_lengths=seq_len
        )
        if is_training:
            tf.summary.histogram('transition', trans)
            tf.summary.image('transition', tf.expand_dims(tf.expand_dims(trans, -1), 0))
    return trans, log_likelihood


def crf_decode(logits, trans, seq_len, idx2tag, is_training, mask=None):
    """
    Decode sequence labelling
    Mask is only used in mutlitask/Adversarial task, where mask indicates prediction for certain task
    """
    with tf.variable_scope('crf_layer'):
        pred_ids, _ = tf.contrib.crf.crf_decode(potentials=logits,
                                                transition_params=trans,
                                                sequence_length=seq_len)
        if is_training:
            pred2str = map2sequence(idx2tag)
            if mask is None:
                tf.summary.text('prediction', pred2str(pred_ids[0, :]))
            else:
                tf.summary.text('prediction', pred2str(tf.boolean_mask(pred_ids, mask, axis=0)[0, :]))
    return pred_ids






if __name__ == '__main__':
    batch_size =2
    max_seq_len= 4
    label_size = 3
    mask = tf.constant([
        [1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]
    ])
    label_ids = tf.constant([
      [1,1,2,0],[2,1,1,0]
    ])
    probs = tf.constant(
        [[[0.1,0.8,0.1],[0.2,0.6,0.2],[0.4,0.5,0.1], [0.4,0.3,0.3]],
         [[0.2,0.7,0.1],[0.3,0.6,0.1],[0.2,0.5,0.3], [0.1,0.3,0.6]]
         ]
    )
    i = 1
    input_prob = tf.gather(probs, tf.cast(i, tf.int32), axis=-1)  # single class probs: batch * max_seq
    input_label = tf.cast(tf.equal(label_ids, i), tf.float32)  # batch * max_seq

