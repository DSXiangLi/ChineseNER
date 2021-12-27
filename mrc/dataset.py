# -*-coding:utf-8 -*-
"""
之前的TFREcord的好处是不用一次读入全部数据, 其实更适合图片类和大型文本任务，这里换成generator，
这里换成一次读入全部数据，然后用generator直接生成dataset
"""

import json
import os
import tensorflow as tf
from data.tokenizer import get_bert_tokenizer

Tag2Query = {
    'LOC': '按照地理位置划分的国家,城市,乡镇,大洲',
    'PER': '人名和虚构的人物形象',
    'ORG': '组织包括公司,政府党派,学校,政府,新闻机构'
}

Tag2Idx = {
    'O': 0,
    'B': 1,
    'I': 2
}

Idx2Tag = dict([(j,i) for i,j in Tag2Idx.items()
                ])

class GeneratorDataset(object):
    def __init__(self, data_dir, batch_size):
        self.data_dir  = data_dir
        self.batch_size = batch_size
        self.data_list = []
        self.samples = []
        self.shapes = {}
        self.types = {}
        self.pads = {}
        self.feature_names = []
        self.label_names = []
        self.tokenizer = get_bert_tokenizer()

    @property
    def n_samples(self):
        return len(self.samples)

    @property
    def steps_per_epoch(self):
        return int(self.n_samples/self.batch_size)

    def load_data(self, file_name):
        # Data Loader for each task
        raise NotImplementedError

    def build_feature(self, *args):
        # Build Feature logic for each task
        raise NotImplementedError

    def build_generator(self):
        for s in self.samples:
            feature = {i: s[i] for i in self.feature_names}
            label = {i: s[i] for i in self.label_names}
            yield feature, label

    def build_serving_proto(self):
        receiver_tensor = {}
        for i in self.feature_names:
            receiver_tensor[i] = tf.placeholder(dtype=self.types[i], shape=[None]+self.shapes[i], name=i)
        return tf.estimator.export.ServingInputReceiver(receiver_tensor, receiver_tensor)

    def build_input_fn(self, is_predict=False):
        def helper():
            shapes = ({i: self.shapes[i] for i in self.feature_names}, {i: self.shapes[i] for i in self.label_names})
            types = ({i: self.types[i] for i in self.feature_names}, {i: self.types[i] for i in self.label_names})
            pads = ({i: self.pads[i] for i in self.feature_names}, {i: self.pads[i] for i in self.label_names})
            dataset = tf.data.Dataset.from_generator(
                lambda: self.build_generator(),
                output_types=types, output_shapes=shapes
            )
            if not is_predict:
                # here we repeat forever and use max_steps in TrainSpec to control n_epochs
                dataset = dataset.shuffle(64). \
                        padded_batch(self.batch_size, shapes, pads).\
                        repeat()
            else:
                dataset = dataset.batch(1)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            return dataset
        return helper


class MRCBIODataset(GeneratorDataset):
    """
    BIO Tagging Schema of MRC Model
    """
    def __init__(self, data_dir, batch_size, max_seq_len, tag2idx):
        super(MRCBIODataset, self).__init__(data_dir, batch_size)
        self.max_seq_len = max_seq_len
        self.tag2idx = tag2idx
        self.tag2query = Tag2Query
        self.define_proto()

    @property
    def label_size(self):
        return len(self.tag2idx)

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
            'label_ids': [None],
        }
        self.types = {
            'input_ids': tf.int32,
            'segment_ids': tf.int32,
            'query_len': tf.int32,
            'text_len': tf.int32,
            'label_ids': tf.int32
        }
        self.pads ={
            'input_ids': self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
            'segment_ids': 1,
            'query_len': 0,
            'text_len': 0,
            'label_ids': 0
        }
        self.feature_names = ['input_ids', 'segment_ids', 'query_len', 'text_len']
        self.label_names = ['label_ids']

    def load_data(self, file_name):
        data_list = []
        with open(os.path.join(self.data_dir, file_name + '_mrc.txt'), 'r') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))
        self.data_list = data_list

    def get_query(self, tag):
        return ['[CLS]'] + list(self.tag2query[tag]) + ['[SEP]']

    def get_label(self, label_list, tag, query_len, text_len):
        """
        根据(start_pos, end_pos)， 用BIO序列标注的方式来生成label
        """
        seq_len = query_len + text_len
        label_ids = [self.tag2idx['O']] * seq_len
        if not label_list:
            return label_ids

        for label in label_list:
            if (label['tag'] == tag) and (label['end_pos'] < text_len):
                #干掉被max_len截断的部分, 前面query_len的部分保持0
                label_ids[label['start_pos']+query_len] = self.tag2idx['B']
                label_ids[(label['start_pos']+query_len+1):(label['end_pos']+query_len+1)] = [self.tag2idx['I']] * (label['end_pos']-label['start_pos'])
        return label_ids

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

            label_ids = self.get_label(data['label'], tag, query_len, text_len)

            sample = {
                'input_ids': input_ids,
                'segment_ids': segment_ids,
                'query_len': query_len,
                'text_len': text_len,
                'label_ids': label_ids,
                # for inference only field, not passed to model
                'text': text,
                'input': input,
                'tag': tag,
                'query': query,
            }
            assert len(label_ids) == len(input_ids), '{}!={}, {}, {}, {}, {}'.format(
                len(label_ids), len(input_ids), tag, text, query_len, text_len)

            samples.append(sample)
        return samples

    def build_feature(self, file_name):
        samples = []
        self.load_data(file_name)
        for data in self.data_list:
            samples += self.build_single_feature(data)
        self.samples = samples


if __name__ == '__main__':
    pipe = MRCBIODataset(data_dir='./data/msra', batch_size=30, max_seq_len=150, tag2idx= Tag2Idx)
    sess = tf.Session()
    pipe.build_feature('train')
    train_input = pipe.build_input_fn(False)
    iterator = tf.data.make_initializable_iterator(train_input())
    sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    features = sess.run(iterator.get_next())

    # import numpy as np
    # for i in range(10):
    #     print(pipe.data_list[i]['label'])
    #     for j in range(3):
    #         print(np.sum(pipe.samples[i*3+j]['span_ids']))
