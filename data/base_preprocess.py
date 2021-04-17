# -*-coding:utf-8 -*-
import os
import pickle
from bert_base.bert import tokenization
import tensorflow as tf
from collections import namedtuple

TrainFeature = namedtuple('TrainFeature', ['tokens','token_ids','mask','segment_ids',
                                           'labels','label_ids','seq_len'])

SoftWord2idx = {
    'B':0,
    'I':1,
    'O':2
}


def get_feature_poroto(max_seq_len):
    feature_proto = {
        'tokens': tf.io.FixedLenFeature([max_seq_len], dtype=tf.string),
        'token_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'mask': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'labels': tf.io.FixedLenFeature([max_seq_len], dtype=tf.string),
        'label_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'seq_len': tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    return feature_proto


class DumpTFRecord(object):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len, tag2idx):
        self.data_dir = data_dir
        self.file_name = file_name
        self.max_seq_len = max_seq_len
        self.tag2idx = tag2idx
        self.tokenizer = tokenization.FullTokenizer(os.path.join(bert_model_dir, "vocab.txt"), do_lower_case=True)
        self.mapping = None

    def read_text(self, filename):
        data = []
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            for line in f:
                data.append(line.strip())
        return data

    def load_data(self):
        raise NotImplementedError()

    @property
    def rename_file(self):
        return self.mapping[self.file_name]

    def format_sequence(self, seq):
        """
        truncate sequence to max_len
        add [CLS], [SEP], [PAD]
        """
        len_seq = len(seq)
        if len_seq > self.max_seq_len - 2:
            seq = seq[:(self.max_seq_len-2)]

        seq = ['[CLS]'] + seq + ['[SEP]']
        seq += ['[PAD]'] * (self.max_seq_len - 2 - len_seq)
        return seq

    def add_softword(self, sentence, tokens):
        """
        add softward ID for each token
        """

    def build_feature(self, sentence, tag):
        tokens = self.tokenizer.tokenize(sentence)

        tags = tag.split(' ')

        if len(tokens) != len(tags):
            print(tag)
            print(sentence)
            print(tokens)
            return None
        tokens = self.format_sequence(tokens)
        labels = self.format_sequence(tags)

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [self.tag2idx[i] for i in labels]
        real_len = sum([i>0for i in label_ids])

        segment_ids = [0] * self.max_seq_len
        mask = [1] * (real_len) + [0] * (self.max_seq_len - real_len)

        assert len(tokens) == self.max_seq_len
        assert len(token_ids) == self.max_seq_len
        assert len(labels) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(mask) == self.max_seq_len
        return TrainFeature(tokens, token_ids, mask, segment_ids, labels, label_ids, real_len)

    @staticmethod
    def string_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(i, encoding='UTF-8') for i in value]))

    @staticmethod
    def int_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def dump_tfrecord(self):
        sentence_list, tag_list = self.load_data()
        n_sample = 0
        n_invalid_sample = 0
        with tf.io.TFRecordWriter(os.path.join(self.data_dir, self.rename_file + '.tfrecord')) as writer:
            for sentence, tag in zip(sentence_list, tag_list):
                feature = self.build_feature(sentence, tag)
                if feature:
                    n_sample+=1
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'tokens': self.string_feature(feature.tokens),
                                'token_ids': self.int_feature(feature.token_ids),
                                'mask': self.int_feature(feature.mask),
                                'segment_ids': self.int_feature(feature.segment_ids),
                                'labels': self.string_feature(feature.labels),
                                'label_ids': self.int_feature(feature.label_ids),
                                'seq_len': self.int_feature(feature.seq_len)
                            }
                        )
                    )
                    writer.write(example.SerializeToString())
                else:
                    n_invalid_sample+=1
        print('Dump {} sample, invalid_sample = {}'.format(n_sample, n_invalid_sample))
        if 'train' in self.file_name:
            self.dump_data_params(n_sample)

    def dump_data_params(self, n_sample):
        """
        Dump data params, which will be used in input_pipe or model_fn later
        """
        params = {
            'n_sample': n_sample,
            'max_seq_len': self.max_seq_len,
            'label_size': len(self.tag2idx),
            'tag2idx': self.tag2idx,
            'idx2tag': dict([(val, key)for key, val in self.tag2idx.items()])
        }
        with open(os.path.join(self.data_dir, 'data_params.pkl'), 'wb') as f:
            pickle.dump(params, f)

