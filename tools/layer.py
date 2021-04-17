# -*-coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from bert_base.bert import modeling
from tools.train_utils import id2sequence
from tools.utils import add_layer_summary


def build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size):
    if cell_type.lower() == 'gru':
        cell_class = tf.nn.rnn_cell.GRUCell
    elif cell_type.lower() == 'lstm':
        cell_class = tf.nn.rnn_cell.LSTMCell
    else:
        raise Exception('Only gru, lstm are supported as cell_type')

    return tf.nn.rnn_cell.MultiRNNCell(
        cells = [ tf.nn.rnn_cell.DropoutWrapper(cell = cell_class(num_units = hidden_units_list[i],
                                                                  activation=activation,
                                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                                  bias_initializer=tf.contrib.layers.xavier_initializer()),
                                                output_keep_prob=keep_prob_list[i],
                                                state_keep_prob=keep_prob_list[i]) for i in range(cell_size) ]
    )


def bilstm(embedding, cell_type, activation, hidden_units_list, keep_prob_list, cell_size, dtype, is_training):
    with tf.variable_scope('bilstm_layer'):
        if not is_training:
            keep_prob_list = len(keep_prob_list) * [1.0]
        fw = build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size)
        bw = build_rnn_cell(cell_type, activation, hidden_units_list, keep_prob_list, cell_size)

        # tuple of 2 : batch_size * max_seq_len * hidden_size
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, embedding, dtype=dtype)

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
            pred2str = id2sequence(idx2tag)
            if mask is None:
                tf.summary.text('prediction', pred2str(pred_ids[0, :]))
            else:
                tf.summary.text('prediction', pred2str(tf.boolean_mask(pred_ids, mask, axis=0)[0, :]))
    return pred_ids


def layer_norm(x):
    """
    layer normalization from Jimmy, apply normalization along feature and apply transformation
    """
    with tf.variable_scope('layer_normalization', reuse=tf.AUTO_REUSE):
        d_model = x.shape.as_list()[-1]
        epsilon = tf.constant(np.finfo(np.float32).eps)
        mean, variance = tf.nn.moments(x, axes=-1, keep_dims=True)
        x = (x - mean)/((variance + epsilon)**0.5) # do layer norm
        add_layer_summary('norm', x)

        kernel = tf.get_variable('norm_kernel', shape=(d_model,), initializer=tf.ones_initializer())
        bias = tf.get_variable('norm_bias', shape=(d_model,),initializer=tf.zeros_initializer())
        x= tf.multiply(kernel, x) +bias
        add_layer_summary('norm_transform', x)
    return x


def add_and_norm_layer(x, sub_layer_x):
    """
    combine Residual connection & layer_norm
    """
    with tf.variable_scope('add_and_norm'):
        x = tf.add(x, sub_layer_x)
        x = layer_norm(x)
    return x


def scaled_dot_product_attention(key, value, query, mask):
    """
    apply dot product attention with mask
    input:
        key: batch_size * key_len * emb_size
        query: batch_size * query_len * emb_size
        value: batch_size * key_len * emb_size
        mask: batch_size * key_len
    output:
        weighted_val: batch_size * query_len * emb_size
    """
    with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
        # scalaed weight matrix : batch_size * query_len * key_len
        dk = tf.cast(key.shape.as_list()[-1], tf.float32)# emb_size
        weight = tf.matmul(query, key, transpose_b=True)/(dk**0.5)

        # apply mask: large negative will become 0 in softmax[mask=0 ignore]
        weight += (1-mask) * (-2**32+1)
        # normalize on axis key_len so that score add up to 1
        weight = tf.nn.softmax(weight, axis=-1)
        tf.summary.image("attention", tf.expand_dims(weight[:1], -1))  # add channel dim
        add_layer_summary('attention', weight)
        # weighted value: batch_size * query_len * emb_size
        weighted_value = tf.matmul(weight, value )

        return weighted_value


def multi_head_attention(key, value, query, mask, params, mode):
    """
    Mutlihead attention with mask
    input:
        key: batch_size * key_len * emb_size
        query: batch_size * query_len * emb_size
        value: batch_size * key_len * emb_size
        mask: batch_size * key_len
    output:
        weighted_val: batch_size * query_len * emb_size
    """
    with tf.variable_scope('multi_head_attention', reuse=tf.AUTO_REUSE):
        d_model = value.shape.as_list()[-1] # emb_size
        # linear projection with dimension unchaangned
        new_key = tf.layers.dense(key, units=d_model, activation=None, name='pre_key_project') # batch_size * key_len * emb_size
        new_value = tf.layers.dense(value, units=d_model, activation=None, name='pre_value_project')
        new_query = tf.layers.dense(query, units=d_model, activation=None, name='pre_query_project')

        # split d_model by num_head and compute attention in parallel
        # (batch_size * num_head) * key_len * (emb_size/num_head)
        new_key = tf.concat(tf.split(new_key, num_or_size_splits=params['num_head'], axis=-1), axis=0)
        new_value = tf.concat(tf.split(new_value, num_or_size_splits=params['num_head'], axis=-1), axis=0)
        new_query = tf.concat(tf.split(new_query, num_or_size_splits=params['num_head'], axis=-1), axis=0)

        # calculate dot-product attention
        weighted_val = scaled_dot_product_attention(new_key, new_value, new_query, tf.tile(mask, [params['num_head'], 1, 1]))

        # concat num_head back
        # (batch_size * num_head) * query_len * (emb_size/num_head) -> batch_size * query_len * emb_size
        weighted_val = tf.concat(tf.split(weighted_val, num_or_size_splits=params['num_head'], axis=0), axis=-1)

        # Linear projection
        weighted_val = tf.layers.dense(weighted_val, units=d_model, activation=None, name='post_linear_project')
        # Do dropout
        weighted_val = tf.layers.dropout(weighted_val, rate=params['dropout_rate'],
                                         training=(mode == tf.estimator.ModeKeys.TRAIN))
        add_layer_summary('raw_multi_head', weighted_val)
        weighted_val = add_and_norm_layer(query, weighted_val)

    return weighted_val


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

