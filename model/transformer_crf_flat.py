# -*-coding:utf-8 -*-
from tools.layer import *
from config import TRAIN_PARAMS
from tools.transformer.modules import *
from tools.transformer.encoder import flat_encoder
from tools.train_utils  import id2sequence

def build_graph(features, labels, params, is_training):
    """
    Use pretrain bert input token embedding as input + bilstm + crf
    """
    input_ids = features['token_ids']
    bichar_ids = features['bichar_ids']
    label_ids = features['label_ids']
    seq_len = features['seq_len']
    lex_len = features['lex_len']
    pos_s = features['pos_s']
    pos_e = features['pos_e']

    if is_training:
        # show position id for debug purpose
        tf.summary.text('pos_s', id2sequence(pos_s[0, :]))
        tf.summary.text('pos_e', id2sequence(pos_e[0, :]))

    with tf.variable_scope('embedding'):
        # Refer to TENER paper we concat char+bichar embedding.
        # batch * max_seq_len * embedding
        lattice_embedding = tf.nn.embedding_lookup(params['lattice_embedding'], input_ids)
        bichar_embedding = tf.nn.embedding_lookup(params['bichar_embedding'], bichar_ids)

        # char+ bichar projection: 取[:seq_len]的纯字符部分， 其余部分mask
        char_mask = tf.sequence_mask(lengths=tf.to_int32(seq_len), maxlen=params['max_seq_len'],
                                dtype=tf.float32)
        embed_char = embedding_project(tf.concat([lattice_embedding, bichar_embedding], axis=-1), params['d_model'])
        add_layer_summary('embed_char', embed_char)
        embed_char = tf.einsum('bld,bl->bld', embed_char, char_mask) # mask all lex_len position
        add_layer_summary('embed_char_mask', embed_char)

        # lattice projection： 取[seq_len:(seq_len+lex_len)]的lattice部分，其余部分mask
        lex_mask = tf.sequence_mask(lengths=tf.to_int32(seq_len+lex_len), maxlen=params['max_seq_len'],
                                    dtype=tf.bool)
        lex_mask = tf.cast(lex_mask ^ tf.cast(char_mask, tf.bool), tf.float32)
        embed_lex = embedding_project(lattice_embedding, params['d_model'])
        add_layer_summary('embed_lex', embed_lex)
        embed_lex = tf.einsum('bld,bl->bld', embed_lex, lex_mask) # mask all lex_len position
        add_layer_summary('embed_lex_mask', embed_lex)

        # Add and dropout
        embedding = embed_lex + embed_char
        embedding = tf.layers.dropout(embedding, seed=1234, rate=params['embedding_dropout'],
                                      training=is_training)

    # In FF use seq_len+lex_len
    transformer_output = flat_encoder(encoder_input=embedding, seq_len=seq_len, lex_len=lex_len,
                                      pos_s=pos_s, pos_e=pos_e, max_raw_seq_len =params['max_raw_seq_len'],
                                      max_seq_len=params['max_seq_len'], encode_layers=params['encode_layers'],
                                      num_head=params['num_head'], dropout_rate=params['dropout_rate'],
                                      ffn_hidden=params['ffn_hidden'], is_training=is_training)
    transformer_output = tf.layers.dropout(transformer_output, seed=1234, rate=params['fc_dropout'],
                                      training=is_training)

    # Drop lattice part for predictioni
    transformer_output = transformer_output[:, :params['max_raw_seq_len'],:]
    logits = tf.layers.dense(transformer_output, units=params['label_size'], activation=None,
                             use_bias=True, name='logits')
    add_layer_summary(logits.name, logits)

    # In CRF and classification only use seq_len out of max_seq_len are used
    label_ids = label_ids[:, :params['max_raw_seq_len']]
    trans, log_likelihood = crf_layer(logits, label_ids, seq_len, params['label_size'], is_training)
    pred_ids = crf_decode(logits, trans, seq_len, params['idx2tag'], is_training)
    crf_loss = tf.reduce_mean(-log_likelihood)

    return crf_loss, pred_ids

# below params from MSRA[smaller params: num_head=5/d_model=200 for people_daily]
TRANSFORMER_PARAMS = {
    'num_head': 8, # giga embedding size is 50, must be divided by 5
    'd_model': 160,  # giga char& bichar embedding dim are small, project to bigger dim
    'ffn_hidden': 480,
    'encode_layers': 2,
    'batch_size': 12,
    'wramup_ratio': 0.1,
    'epochs': 100
}

TRAIN_PARAMS.update(TRANSFORMER_PARAMS)
TRAIN_PARAMS.update({
    'lr': 0.001,
    'decay_rate': 0.95,  # lr * decay_rate ^ (global_step / train_steps_per_epoch)
    'embedding_dropout': 0.3,
    'fc_dropout': 0.2,
    'dropout_rate': 0.2, # used in transformer sublayer dropout
    'early_stop_ratio': 2 # stop after no improvement after 1.5 epochs
})