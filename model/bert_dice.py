# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from tools.loss import dice_loss
from config import TRAIN_PARAMS



def build_graph(features, labels,  params, is_training):
    """
    pretrain Bert model output + CRF Layer
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']
    input_mask = features['mask']
    segment_ids = features['segment_ids']
    seq_len = features['seq_len']
    embedding = pretrain_bert_embedding(input_ids, input_mask, segment_ids, params['pretrain_dir'],
                                        params['embedding_dropout'], is_training)
    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    logits = tf.layers.dense(embedding, units=params['label_size'], activation=None,
                             use_bias=True, name='logits')
    add_layer_summary(logits.name, logits)

    loss = dice_loss(logits, label_ids, seq_len, params['idx2tag'], params['max_seq_len'], params['alpha'], params['gamma'])
    pred_ids = tf.argmax(logits, axis=-1)
    return loss , pred_ids

TRAIN_PARAMS.update({
    'diff_lr_times': {'crf': 500,  'logit': 500},
    'alpha': 1 ,
    'gamma': 1
})

