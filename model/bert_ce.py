# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.loss import cross_entropy_loss
from tools.utils import add_layer_summary
from config import TRAIN_PARAMS


def build_graph(features, labels, params, is_training):
    """
    pretrain Bert model output + cross-entropy loss
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

    loss = cross_entropy_loss(logits, label_ids, seq_len,
                                        params['label_size'], params['max_seq_len'], params['dtype'])
    pred_ids = tf.argmax(logits, axis=-1)  # batch * max_seq
    if is_training:
        pred2str = map2sequence(params['idx2tag'])
        tf.summary.text('prediction', pred2str(pred_ids[0, :]))

    return loss, pred_ids


TRAIN_PARAMS.update({
    'diff_lr_times': {'logit': 500}
})
