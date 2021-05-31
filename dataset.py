# -*-coding:utf-8 -*-

import os
import pickle
import tensorflow as tf
import numpy as np
from data.base_preprocess import get_feature_poroto
from data.word_enhance import SoftWord, SoftLexicon, ExSoftWord, WordEnhanceMethod


class NerDataset(object):
    def __init__(self, data_dir, batch_size, epoch_size, word_enhance=None):
        assert word_enhance in [None] + WordEnhanceMethod, 'word_enhance must in {}'.format(','.join(WordEnhanceMethod))
        self.word_enhance = word_enhance
        self.surfix = '' if word_enhance is None else word_enhance #use to distinguish tfrecord with/out word enhance
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self._params = None
        self.init_params()
        self.proto = get_feature_poroto(self.params['max_seq_len'], word_enhance)

    def parser(self, line):
        features = tf.parse_single_example(line, features=self.proto)
        features['token_ids'] = tf.cast(features['token_ids'], tf.int32)
        features['mask'] = tf.cast(features['mask'], tf.int32)
        features['segment_ids'] = tf.cast(features['segment_ids'], tf.int32)
        features['label_ids'] = tf.cast(features['label_ids'], tf.int32)
        features['seq_len'] = tf.squeeze(tf.cast(features['seq_len'], tf.int32))
        # adjust parser when word enhance method is used
        if self.word_enhance == SoftWord:
            features['softword_ids'] = tf.cast(features['softword_ids'], tf.int32)
        elif self.word_enhance == ExSoftWord:
            # cast to float and reshape to original 2 dimension
            features['ex_softword_ids'] = tf.reshape(
                tf.cast(features['ex_softword_ids'], tf.float32), [-1, self.params['word_enhance_dim']])
        elif self.word_enhance == SoftLexicon:
            features['softlexicon_ids'] = tf.reshape(
                tf.cast(features['softlexicon_ids'], tf.int32), [-1, self.params['word_enhance_dim']])
            features['softlexicon_weights'] = tf.reshape(
                tf.cast(features['softlexicon_weights'], tf.float32), [-1, self.params['word_enhance_dim']])
        return features

    def build_input_fn(self, file_name, is_predict=0, unbatch=False):
        def input_fn():
            dataset = tf.data.TFRecordDataset(
                os.path.join(self.data_dir, '_'.join(filter(None, [file_name, self.surfix])) + '.tfrecord')). \
                map(lambda x: self.parser(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if not is_predict:
                dataset = dataset.shuffle(64). \
                    repeat(self.epoch_size)

            if not unbatch:
                # For performace issue, not to use unbatch in mutitask
                dataset = dataset. \
                    batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            return dataset
        return input_fn

    def init_params(self):
        """
        Inherit max_seq_len, label_size, n_sample from data_preprocess per dataset
        """
        with open(os.path.join(self.data_dir, '_'.join(filter(None, [self.surfix, 'data_params.pkl']))), 'rb') as f:
            self._params = pickle.load(f)
        self._params['step_per_epoch'] = int(self._params['n_sample']/self.batch_size)
        self._params['num_train_steps'] = int(self.epoch_size * self._params['step_per_epoch'])

    @property
    def params(self):
        return self._params


class MultiDataset(object):
    """
    Used for Multi-Task & Adversarial task. Each batch will include samples from all tasks with same size
    For now only 2 task are supported
    """
    def __init__(self, root_dir, data_list, batch_size, epoch_size):
        self._params = {}
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.data_list = data_list
        self.dataset_dict = dict([(dir, NerDataset(os.path.join(root_dir, dir), batch_size, epoch_size)) \
                                  for dir in data_list])
        self.init_params()

    def add_discriminator(self, features, task_id):
        features['task_ids'] = np.ones_like(features['token_ids']) * task_id
        return features

    def build_input_fn(self, file_name):
        def input_fn():
            dataset_list = [dataset.build_input_fn(file_name, is_predict=0, unbatch=True)().\
                                map(lambda x: self.add_discriminator(x, i))
                            for i, dataset in enumerate(self.dataset_dict.values())]
            choice_dataset = tf.data.Dataset.range(2).repeat()
            dataset = tf.contrib.data.choose_from_datasets(dataset_list,
                                                           choice_dataset)
            dataset = dataset.repeat(self.epoch_size).batch(self.batch_size)
            return dataset

        return input_fn

    def build_predict_fn(self, data):
        """
        For prediction, return input_fn for data each time
        """
        def input_fn():
            dataset = self.dataset_dict[data]
            dataset = dataset.build_input_fn('predict', is_predict=True, unbatch=True)(). \
                map(lambda x: self.add_discriminator(x, self.data_list.index(data)))
            dataset = dataset.batch(self.batch_size)
            return dataset
        return input_fn

    def init_params(self):
        for data, dataset in self.dataset_dict.items():
            self._params.update({
                data: dataset.params
            })
        # use smaller step_per_epoch between 2 dataset
        self._params['step_per_epoch'] = int(max([p['step_per_epoch'] for p in self._params.values()]))
        self._params['num_train_steps'] = int(self.epoch_size * self._params['step_per_epoch'])
        self._params['task_list'] = self.data_list

    @property
    def params(self):
        return self._params


if __name__ == '__main__':
    prep = NerDataset('./data/msra', 1, 10, word_enhance=SoftLexicon)
    train_input = prep.build_input_fn('train')

    sess = tf.Session()
    iterator = tf.data.make_initializable_iterator(train_input())
    sess.run( iterator.initializer )
    sess.run( tf.tables_initializer() )
    sess.run( tf.global_variables_initializer() )
    features = sess.run( iterator.get_next() )
    print(features)

    prep = MultiDataset('./data', ['msra','people_daily'], 4 , 2)
    train_input = prep.build_input_fn('train')
    sess = tf.Session()
    iterator = tf.data.make_initializable_iterator(train_input())
    sess.run( iterator.initializer )
    sess.run( tf.tables_initializer() )
    sess.run( tf.global_variables_initializer() )
    features = sess.run( iterator.get_next() )
    print(features['labels'])
    print(features['task_ids'])


    prep = MultiDataset('./data', ['msra','people_daily'], 4 , 2)
    train_input = prep.build_predict_fn('msra')
    sess = tf.Session()
    iterator = tf.data.make_initializable_iterator(train_input())
    sess.run( iterator.initializer )
    sess.run( tf.tables_initializer() )
    sess.run( tf.global_variables_initializer() )
    features = sess.run( iterator.get_next() )
    print(features['labels'])
    print(features['task_ids'])