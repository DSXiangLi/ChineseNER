# -*-coding:utf-8 -*-
from tools.train_utils import *
from tools.layer import *
from tools.utils import add_layer_summary
from mrc.model import cross_entropy_loss_mask
from mrc.dataset import GeneratorDataset, Tag2Query
import numpy as np
import json

class MRCDataset(GeneratorDataset):
    def __init__(self, data_dir, batch_size, max_seq_len):
        super(MRCDataset, self).__init__(data_dir, batch_size)
        self.max_seq_len = max_seq_len
        self.tag2query = Tag2Query
        self.define_proto()

    def define_proto(self):
        """
        Define shape and dtype for feature and label, will be used in dataset and inference
        """
        self.shapes = {
            'input': [None],
            'input_ids': [None],
            'segment_ids': [None],
            'query_len': [],
            'text_len': [],
            'start_ids': [None],
            'end_ids': [None],
            'span_ids': [None],
        }
        self.types = {
            'input_ids': tf.int32,
            'segment_ids': tf.int32,
            'query_len': tf.int32,
            'text_len': tf.int32,
            'start_ids': tf.int32,
            'end_ids': tf.int32,
            'span_ids': tf.int32
        }
        self.pads ={
            'input_ids': self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
            'segment_ids': 1,
            'query_len': 0,
            'text_len': 0,
            'start_ids': 0,
            'end_ids': 0,
            'span_ids': 0
        }
        self.feature_names = ['input_ids', 'segment_ids', 'query_len', 'text_len']
        self.label_names = ['start_ids', 'end_ids', 'span_ids']

    def load_data(self, file_name):
        data_list = []
        with open(os.path.join(self.data_dir, file_name + '_mrc.txt'), 'r') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))
        self.data_list = data_list

    def get_query(self, tag):
        return ['[CLS]'] + list(self.tag2query[tag]) + ['[SEP]']

    @staticmethod
    def get_label(label_list, tag, query_len, text_len):
        """
        根据(start_pos, end_pos)， 生成某个实体类型的start_ids, end_ids
        """
        seq_len = query_len + text_len
        start_ids = [0] * seq_len
        end_ids = [0] * seq_len
        span_ids = np.zeros(shape=(seq_len, seq_len))
        if not label_list :
            return start_ids, end_ids, span_ids

        for label in label_list:
            if (label['tag'] == tag) and (label['end_pos'] < text_len):
                #干掉被max_len截断的部分, 前面query_len的部分保持0
                start_ids[label['start_pos']+query_len] = 1
                end_ids[label['end_pos']+query_len] = 1
                span_ids[label['start_pos']+query_len][label['end_pos']+query_len] = 1
        return start_ids, end_ids, span_ids

    def build_single_feature(self, data):
        """
        data format defined in convert2mrc
        """
        samples = []
        for tag in self.tag2query:
            # get query for tag and pad with CLS and SEP
            query = self.get_query(tag)
            query_len = len(query)

            # chunk to text to max_len - query len
            text = data['title'].split()
            text = text[: self.max_seq_len - query_len]
            text_len = len(text)
            input = query+text
            input_ids = self.tokenizer.convert_tokens_to_ids(input) # 这里直接对char进行convert了，会增加OOV但是会避免BPE分割问题
            segment_ids = [0] * query_len + [1] * text_len

            start_ids, end_ids, span_ids = self.get_label(data['label'], tag, query_len, text_len)

            sample = {
                'input_ids': input_ids,
                'segment_ids': segment_ids,
                'query_len': query_len,
                'text_len': text_len,
                'start_ids': start_ids,
                'end_ids': end_ids,
                'span_ids': span_ids.reshape(-1).tolist(), # flatten to dim1
                # for inference only field, not passed to model
                'text': text,
                'input': input,
                'tag': tag,
                'query': query,
            }
            samples.append(sample)
        return samples

    def build_feature(self, file_name):
        samples = []
        self.load_data(file_name)
        for data in self.data_list:
            samples += self.build_single_feature(data)
        self.samples = samples


def span_graph(features, labels,  params, is_training):
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

    # start & end loss
    with tf.variable_scope('start_end_prediction'):
        start_logits = tf.layers.dense(embedding, units=2, activation=None, use_bias=True, name='start_logits')
        end_logits = tf.layers.dense(embedding, units=2, activation=None, use_bias=True, name='end_logits')

        start_probs = tf.nn.softmax(start_logits, axis=-1)
        end_probs =tf.nn.softmax(end_logits, axis=-1)

        add_layer_summary('start_probs', start_probs)
        add_layer_summary('end_probs', end_probs)

    with tf.variable_scope('span_prediction'):
        # span_hidden: [batch, seq_len, hidden] -> [batch, seq_len, seq_len, hidden]
        span_hidden = tf.concat([
            tf.tile(tf.expand_dims(embedding, 1), [1, max_seq_len, 1, 1]),
            tf.tile(tf.expand_dims(embedding, 2), [1, 1, max_seq_len, 1])
        ], axis=-1)

        # span_logits: [batch, seq_len, seq_len, 2] -> [batch, seq_len * seq_len, 2]
        span_hidden = tf.layers.dense(span_hidden, units=params['span_hidden_size'], activation=gelu,
                                      use_bias=True, name='span_hideen')
        span_hidden = tf.layers.dropout(span_hidden, rate=params['span_dropout'], seed=1234, training=is_training)
        span_logits = tf.layers.dense(span_hidden, units=params['label_size'], activation=None,
                                      use_bias=True, name='span_logits')

        span_logits = tf.reshape(span_logits, [-1, tf.multiply(max_seq_len, max_seq_len), params['label_size']])
        span_probs = tf.nn.softmax(span_logits, axis=-1)
        add_layer_summary('span_probs', span_probs)

    with tf.variable_scope('mask'):
        # Generate Mask for text
        query_mask = tf.cast(tf.sequence_mask(query_len, maxlen=max_seq_len), tf.float32)
        text_mask = tf.cast(input_mask, tf.float32) - query_mask

        add_layer_summary('query_mask', query_mask)
        add_layer_summary('text_mask', text_mask)
        # span_mask: [batch, seq_len] -> [batch, seq_len, seq_len] -> [batch, seq_len * seq_len ]
        span_mask = tf.multiply(tf.tile(tf.expand_dims(text_mask, -1), [1, 1, max_seq_len]),
                                tf.tile(tf.expand_dims(text_mask, -2), [1, max_seq_len, 1]))
        span_mask = tf.linalg.band_part(span_mask, 0, -1)  # 只保留下半矩阵，因为start <= end
        tf.summary.image('span_mask',tf.expand_dims(tf.expand_dims(tf.cast(span_mask[0,:,:], tf.float32), axis=-1),0))
        tf.summary.scalar('span_mask', tf.reduce_sum(span_mask))

    if labels is None:
        return None, start_probs, end_probs, span_probs, text_mask, span_mask

    with tf.variable_scope('span_candidates'):
        # 因为Span Loss非常unbalance,所以在实现中只选取start/end的pred/label中为1的部分来计算span loss，其他部分都过滤掉
        span_candidates = tf.logical_or(tf.tile(tf.expand_dims(start_probs > 0.5, -1), [1, 1, max_seq_len]) &\
                                        tf.tile(tf.expand_dims(end_probs > 0.5, -2), [1, max_seq_len, 1]),
                                        tf.tile(tf.expand_dims(labels['start_ids']>0, -1), [1, 1, max_seq_len]) & \
                                        tf.tile(tf.expand_dims(labels['end_ids']>0, -2), [1, max_seq_len, 1]))
        tf.summary.image('span_candidates', tf.expand_dims(tf.expand_dims(tf.cast(span_candidates[0, :, :], tf.float32), axis=-1), 0))
        span_mask_new = tf.multiply(span_mask, tf.cast(span_candidates, tf.float32))
        span_mask_new = tf.reshape(span_mask_new, [-1, tf.multiply(max_seq_len, max_seq_len)])

        tf.summary.scalar('span_mask_new', tf.reduce_sum(span_mask_new))

    with tf.variable_scope('loss'):
        start_loss = cross_entropy_loss_mask(start_logits, labels['start_ids'], text_mask)
        end_loss = cross_entropy_loss_mask(end_logits, labels['end_ids'], text_mask)
        span_loss = cross_entropy_loss_mask(span_logits, labels['span_ids'], span_mask_new)
        tf.summary.scalar('start_loss', start_loss)
        tf.summary.scalar('end_loss', end_loss)
        tf.summary.scalar('span_loss', span_loss)

        loss = params['start_weight'] * start_loss + params['end_weight'] * end_loss + params['span_weight'] * span_loss

    return loss, start_probs, end_probs, span_probs, text_mask, span_mask_new



def span_model_fn():
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        loss, start_probs, end_probs, span_probs, text_mask, span_mask = span_graph(features, labels, params, is_training)

        if is_training:
            train_op = bert_train_op(loss, params['lr'], params['num_train_steps'],
                                    params['warmup_ratio'], params['diff_lr_times'], True)
            spec = tf.estimator.EstimatorSpec(mode, loss=loss,
                                              train_op=train_op,
                                              training_hooks=[get_log_hook(loss, params['log_steps'])])
        elif mode == tf.estimator.ModeKeys.EVAL:
            metric_op = {}
            metric_op.update(calc_metrics(labels['start_ids'], start_probs, text_mask, 'start'))
            metric_op.update(calc_metrics(labels['end_ids'], end_probs, text_mask, 'end'))
            metric_op.update(calc_metrics(labels['span_ids'], span_probs, span_mask, 'span'))
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_op)
        else:
            spec = tf.estimator.EstimatorSpec(mode, predictions={'start_probs': start_probs,
                                                                 'end_probs': end_probs,
                                                                 'span_probs': span_probs,
                                                                 'span_mask': span_mask,
                                                                 'text_mask': text_mask})
        return spec
    return model_fn



def span_alignment(start_prob, end_prob):
    """
    计算start_porb, end_prob的一直结果
    """
    seq_len = np.shape(start_prob)[0]
    start_prob = start_prob > 0.5
    end_prob = end_prob > 0.5

    match_prob = np.tile(np.expand_dims(start_prob, 1), [1, seq_len]) & \
                 np.tile(np.expand_dims(end_prob, 0), [seq_len,1])

    return match_prob


def entity_extract(prediction, token, tag):
    """
    输入MRC模型预测，实体类型，原始文本返回抽取的实体
    prediction: raw prediction from MRC model, each has field {'start_preds','end_preds','span_preds','span_mask'}
    sentence: [query tokens] + [text tokens]
    tag: entity type
    """
    max_seq_len = np.shape(prediction['start_preds'])[0]

    span_mask = np.multiply(np.tile(np.expand_dims(prediction['text_mask'], -2), [1, max_seq_len, 1]),
                            np.tile(np.expand_dims(prediction['text_mask'], -1), [1, 1, max_seq_len]))
    span_mask = np.astype(np.linalg.band_part(span_mask, -1, 0) , bool) # 只保留下半矩阵，因为start < end

    match_probs = span_alignment(prediction['start_preds'], prediction['end_preds'])

    match_probs = np.multiply(match_probs, span_mask)

    pos = np.where(match_probs >0)

    entities = []
    for i in pos:
        entities.append(''.join(token[i[0]:i[1]]))

    return {tag:entities}