# -*-coding:utf-8 -*-

"""
    Bert MLM for Paraphase Augmentation
    Default to Bert Base WWM Model
"""

import os
import random
import numpy as np
import tensorflow as tf

from bert_base.bert import modeling
from tools.utils import build_estimator
from config import RUN_CONFIG
from tools.train_utils import load_bert_checkpoint
from data.people_daily_augment.augmentation import AugHandler
from tools.fast_predict import FastPredict
from data.tokenizer import get_bert_tokenizer

MAX_SEQ_LEN = 512
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
    # Only Extract MASK input
    input_tensor = tf.gather(input_tensor, positions, axis=1)

    with tf.variable_scope("cls/predictions"):
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                  bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    return log_probs


def build_graph(features, params, is_training):
    input_ids = features['input_ids']
    segment_ids = features['segment_ids']
    seq_len = features['seq_len']
    maxlen = tf.reduce_max(seq_len)
    input_mask = tf.sequence_mask(seq_len, maxlen=maxlen)
    mask_pos = features['mask_pos']

    bert_config = modeling.BertConfig.from_json_file(os.path.join(params['pretrain_dir'], 'bert_config.json'))
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
    probs = get_masked_lm_output(bert_config, embedding, bert_model.get_embedding_table(), mask_pos)
    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    return probs


def build_model_fn():
    def model_fn(features, labels, params, mode):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        probs = build_graph(features, params, is_training)
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode, predictions={'prob': probs})
            return spec
    return model_fn


def build_input_fn(generator):
    def helper():
        dataset = tf.data.Dataset.from_generator(
             generator,
             output_shapes={'input_ids':[None],
                             'segment_ids':[None],
                             'seq_len':[],
                             'mask_pos':[None]},
             output_types={'input_ids': tf.int32,
                         'segment_ids': tf.int32,
                         'seq_len': tf.int32,
                         'mask_pos': tf.int32}).batch(1)
        return dataset
    return helper


estimator = build_estimator({'ckpt_dir': None,
                             'pretrain_dir': './pretrain_model/ch_google',
                             'warm_start': False},
                             None,
                            build_model_fn(), True,
                            RUN_CONFIG)

class MlmSR(AugHandler):
    def __init__(self, max_sample, change_rate):
        super(MlmSR, self).__init__(max_sample, change_rate)
        self.max_seq_len = MAX_SEQ_LEN
        self.tokenizer = get_bert_tokenizer()
        self.available_pos = [] #可以被mask的位置/会剔除存在实体的位置
        self.fp = FastPredict(estimator, build_input_fn)

    def _mask_token(self, tokens):
        n = len(tokens)
        n_mask = max(int(n * self.change_rate), 1)
        if len(self.available_pos) < n_mask *2 :
            raise ValueError('Availble Pos < N_mask*2')
        mask_pos = random.sample(self.available_pos, n_mask)
        for i in mask_pos:
           tokens[i] = '[MASK]'
        return tokens

    def gen_available_pos(self, labels):
        """
        过滤所有的实体位置，只对label=O的部分去做MASK，避免改变实体标签
        """
        labels = labels[:(self.max_seq_len-2)]
        self.available_pos = []
        for i in range(len(labels)):
            if labels[i]=='O':
                self.available_pos.append(i)

    def build_single_feature(self, tokens, labels):
        self.gen_available_pos(labels)
        tokens = self._mask_token(tokens[: self.max_seq_len-2])
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        seq_len = len(tokens)
        mask_pos = [i for i, j in enumerate(tokens) if j == '[MASK]']
        return {
            'input_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'segment_ids': [0] * seq_len,
            'seq_len': seq_len,
            'mask_pos': mask_pos
        }

    def decode_prediction(self, probs):
        probs = np.squeeze(probs, axis=0)
        tokenid = np.argmax(probs, axis=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(tokenid)
        return tokens

    def gen_single_sample(self, sentence, label):
        """
        Sentence and label are ' ' Joined str
        """
        tokens = sentence.split()

        labels = label.split()

        try:
            feature = self.build_single_feature(tokens, labels )
        except ValueError:
            return sentence, label

        pred = self.fp.stream_predict(feature)

        pred_tokens = self.decode_prediction(pred['prob'])

        for i, pos in enumerate(feature['mask_pos']):
            tokens[pos-1] = pred_tokens[i]

        return ' '.join(tokens), label


if __name__ == '__main__':
    sentence= '海 钓 比 赛 地 点 在 厦 门 与 金 门 之 间 的 海 域 。'
    label = 'O O O O O O O B-LOC I-LOC O B-LOC I-LOC O O O O O O'

    mlm = MlmSR(3, 0.1)
    res = mlm.gen_single_sample(sentence, label)
    print(res)
