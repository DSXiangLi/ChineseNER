# -*-coding:utf-8 -*-
import os
import pickle
from bert_base.bert import tokenization
import tensorflow as tf
from itertools import chain
from functools import partial
import importlib
from data.word_enhance import (align_with_token, build_softword, build_ex_softword, build_soft_lexicon,
                               WordEnhanceMethod, SoftWord, SoftLexicon, ExSoftWord,
                               Soft2Idx, MaxLexiconLen, model_init,
                               postproc_soft_lexicon, prebuild_weight)


class TrainFeature:
    def __init__(self, tokens, token_ids, mask,
                 segment_ids, labels, label_ids, seq_len):
        self.tokens = tokens
        self.token_ids = token_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.label_ids = label_ids
        self.seq_len = seq_len


def get_feature_poroto(max_seq_len, word_enhance=None):
    assert word_enhance in [None]+WordEnhanceMethod, 'word_enhance must in {}'.format(','.join(WordEnhanceMethod))

    feature_proto = {
        'tokens': tf.io.FixedLenFeature([max_seq_len], dtype=tf.string),
        'token_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'mask': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'segment_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'labels': tf.io.FixedLenFeature([max_seq_len], dtype=tf.string),
        'label_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64),
        'seq_len': tf.io.FixedLenFeature([], dtype=tf.int64)
    }

    if word_enhance is None:
        pass # basic tf proto without word enhancement
    elif word_enhance == SoftWord:
        feature_proto.update({
            'softword_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64)
        })
    elif word_enhance == ExSoftWord:
        feature_proto.update({
            'ex_softword_ids': tf.io.FixedLenFeature([max_seq_len*len(Soft2Idx)], dtype=tf.int64)
        })
    else:
        feature_proto.update({
            'softlexicon_ids': tf.io.FixedLenFeature([max_seq_len*len(Soft2Idx)*MaxLexiconLen], dtype=tf.int64),
            'softlexicon_weights': tf.io.FixedLenFeature([max_seq_len*len(Soft2Idx)*MaxLexiconLen], dtype=tf.float32)
        })

    return feature_proto


def get_instance(data_dir, file_name, bert_model_dir,
                 max_seq_len, tag2idx, mapping, load_data_func, word_enhance=None, **kwargs):
    # Init instance with different cls, and dynamic add data specific load_data func to instance
    assert word_enhance in [None]+WordEnhanceMethod, 'word_enhance must in {}'.format(','.join(WordEnhanceMethod))

    if word_enhance is None:
        cls = BasicTFRecord
    else:
        model_init() # run lazy variable loading
        if word_enhance == SoftWord:
            cls = SoftwordTFRecord
        elif word_enhance == ExSoftWord:
            cls = ExSoftwordTFRecord
        else:
            cls = SoftLexiconTFRecord

    instance = cls(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func, **kwargs)
    return instance


def read_text(data_dir, filename):
    data = []
    with open(os.path.join(data_dir, filename), 'r') as f:
        for line in f:
            data.append(line.strip())
    return data


class BasicTFRecord(object):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func):
        self.data_dir = data_dir
        self.file_name = file_name
        self.max_seq_len = max_seq_len
        self.tag2idx = tag2idx
        self.tokenizer = tokenization.FullTokenizer(os.path.join(bert_model_dir, "vocab.txt"), do_lower_case=True)
        self.mapping = mapping
        self.surfix = ''# used to distinguish tf record and data params
        self.load_data = load_data_func # pass in load_data_func given differnt dataset
        self.sentence_list, self.tag_list = self.load_data(self.data_dir, self.file_name)

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

    def build_feature(self, sentence, tag):
        tokens = self.tokenizer.tokenize(sentence)

        tags = tag.split(' ')

        assert len(tokens)==len(tags), '{}!={} n_token!=n_tag'.format(len(tokens), len(tags))
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

    @staticmethod
    def float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def build_tf_feature(self, feature):
        tf_feature = {
            'tokens': self.string_feature(feature.tokens),
            'token_ids': self.int_feature(feature.token_ids),
            'mask': self.int_feature(feature.mask),
            'segment_ids': self.int_feature(feature.segment_ids),
            'labels': self.string_feature(feature.labels),
            'label_ids': self.int_feature(feature.label_ids),
            'seq_len': self.int_feature(feature.seq_len)
        }
        return tf_feature

    def build_data_params(self, n_sample):
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
        return params

    def dump_tfrecord(self):
        n_sample = 0
        n_invalid_sample = 0
        with tf.io.TFRecordWriter(os.path.join(self.data_dir, '_'.join(filter(None, [self.rename_file, self.surfix])) + '.tfrecord')) as writer:
            for sentence, tag in zip(self.sentence_list, self.tag_list):
                try:
                    feature = self.build_feature(sentence, tag)
                    n_sample+=1
                    example = tf.train.Example(
                        features=tf.train.Features(feature=self.build_tf_feature(feature))
                    )
                    writer.write(example.SerializeToString())
                except Exception as e:
                    print(e)
                    n_invalid_sample+=1

        print('Dump {} sample, invalid_sample = {}'.format(n_sample, n_invalid_sample))
        if 'train' in self.file_name:
            params = self.build_data_params(n_sample)
            with open(os.path.join(self.data_dir, '_'.join(filter(None, [self.surfix, 'data_params.pkl']))), 'wb') as f:
                pickle.dump(params, f)

####################################
# Following are word enhance class #
####################################


class SoftwordTFRecord(BasicTFRecord):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func):
        super(SoftwordTFRecord, self).__init__(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func)
        self.surfix = SoftWord # surfix = word enhance method
        self.soft2idx = Soft2Idx

    def format_soft_seq(self, seq):
        len_seq = len(seq)
        if len_seq > self.max_seq_len - 2:
            seq = seq[:(self.max_seq_len-2)]

        seq = [self.soft2idx['None']] + seq + [self.soft2idx['None']]
        seq += [self.soft2idx['None']] * (self.max_seq_len - 2 - len_seq)
        return seq

    def build_feature(self, sentence, tag):
        """
        add B/M/E/S labels for each token, using jieba word cut and do label encoding
        softword_ids: max_seq_len * 1
        """
        train_feature = super().build_feature(sentence, tag)
        softword_ids = build_softword(sentence, False)
        softword_ids = align_with_token(softword_ids, train_feature.tokens, word_enhance=self.surfix)
        train_feature.softword_ids = self.format_soft_seq(softword_ids)
        return train_feature

    def build_tf_feature(self, feature):
        tf_feature = super().build_tf_feature(feature)
        tf_feature['softword_ids'] = self.int_feature(feature.softword_ids)
        return tf_feature

    def build_data_params(self, n_sample):
        params = super().build_data_params(n_sample)
        params['soft2idx'] = self.soft2idx
        params['word_enhance_dim'] = len(self.soft2idx) # number of word enhance label encoder
        return params


class ExSoftwordTFRecord(BasicTFRecord):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func):
        super(ExSoftwordTFRecord, self).__init__(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func)
        self.surfix = ExSoftWord
        self.soft2idx = Soft2Idx

    def format_soft_seq(self, seq):
        len_seq = len(seq)
        if len_seq > self.max_seq_len - 2:
            seq = seq[:(self.max_seq_len-2)]

        default_one_hot = [0] * len(self.soft2idx)
        default_one_hot[self.soft2idx['None']] = 1 #[0,0,0,0,1] for cls,sep,pad
        seq = [default_one_hot] + seq + [default_one_hot]
        seq += [default_one_hot] * (self.max_seq_len - 2 - len_seq)
        return seq

    def build_feature(self, sentence, tag):
        """
        getting all possible B/M/E/S for each token and do one-hot encoding
        ex_softword_ids: max_seq_len * 5
        """
        train_feature = super().build_feature(sentence, tag)
        ex_softword_ids = build_ex_softword(sentence, False)
        ex_softword_ids = align_with_token(ex_softword_ids, train_feature.tokens, word_enhance=self.surfix)
        train_feature.ex_softword_ids = self.format_soft_seq(ex_softword_ids)
        return train_feature

    def build_tf_feature(self, feature):
        tf_feature = super().build_tf_feature(feature)
        ## flatten to 1 dimension
        tf_feature['ex_softword_ids'] = self.int_feature(list(chain(*feature.ex_softword_ids)))
        return tf_feature

    def build_data_params(self, n_sample):
        params = super().build_data_params(n_sample)
        params['soft2idx'] = self.soft2idx
        params['word_enhance_dim'] = len(self.soft2idx) # dimension of one-hot word enhance vector
        return params


class SoftLexiconTFRecord(BasicTFRecord):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func, default_weight=True):
        super(SoftLexiconTFRecord, self).__init__(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx, mapping, load_data_func)
        self.surfix = SoftLexicon # surfix = word enhance method
        self.soft2idx=Soft2Idx
        model_init() # Only used when cls is init outside get_instance
        self.default_weight = default_weight # True will use pretrain model vocabFreq, otherwise calculate by input data
        self.postproc_func = self.init_weight()

    def init_weight(self):
        if self.default_weight:
            print('Using default vocab frequency from pretrain model')
            vocabfreq = getattr(importlib.import_module('data.word_enhance'), 'VocabFreq')
            return partial(postproc_soft_lexicon, vocabfreq=vocabfreq)
        # calculate weight for softlexion using training data
        if self.rename_file == 'train':
            prebuild_weight(self.data_dir, self.sentence_list)
        # load vocab frequency
        with open(os.path.join(self.data_dir,'lexicon_weight.pkl'), 'rb') as f:
            self.vocabfreq = pickle.load(f)
        return partial(postproc_soft_lexicon, vocabfreq=self.vocabfreq)

    @property
    def vocab2idx(self):
        # use lazy import and make sure model_init run ahead
        vocab2idx = getattr(importlib.import_module('data.word_enhance'),'Vocab2IDX')
        assert vocab2idx, 'Soft2Idx is None, to use word enhance cls, run model_init first'
        return vocab2idx

    def format_soft_seq(self, seq, type='ids'):
        len_seq = len(seq)
        if len_seq > self.max_seq_len - 2:
            seq = seq[:(self.max_seq_len-2)]

        if type =='weight':
            default_encoding = [0.0] * len(seq[0])
        else:
            default_encoding = [0] * len(seq[0])

        seq = [default_encoding] + seq + [default_encoding]
        seq += [default_encoding] * (self.max_seq_len - 2 - len_seq)
        return seq

    def build_feature(self, sentence, tag):
        """
        getting all possible soft lexicon, truncate/pad to max len and calculate normalized weight
        softlexicon_ids: max_seq_len * (4 * MaxLexiconLen)
        softlexicon_weight: same shape as ids
        """
        train_feature = super().build_feature(sentence, tag)
        soft_lexicon = build_soft_lexicon(sentence, False)
        soft_lexicon = align_with_token(soft_lexicon, train_feature.tokens, word_enhance=self.surfix)
        ids, weights = self.postproc_func(soft_lexicon)
        train_feature.softlexicon_ids = self.format_soft_seq(ids)
        train_feature.softlexicon_weights = self.format_soft_seq(weights, type='weight')
        return train_feature

    def build_tf_feature(self, feature):
        tf_feature = super().build_tf_feature(feature)
        # flatten to 1 dimension
        tf_feature['softlexicon_ids'] = self.int_feature(list(chain(*feature.softlexicon_ids)))
        tf_feature['softlexicon_weights'] = self.float_feature(list(chain(*feature.softlexicon_weights)))
        return tf_feature

    def build_data_params(self, n_sample):
        # pass in pretrain model vocab Embedding
        params = super().build_data_params(n_sample)
        params['soft2idx'] = self.soft2idx
        params['vocab2idx'] = self.vocab2idx
        params['word_enhance_dim'] = len(self.soft2idx) # B/M/E/S/None * max_lexicon_len
        params['max_lexicon_len'] = MaxLexiconLen
        params['word_embedding'] = getattr(importlib.import_module('data.word_enhance'), 'VocabEmbedding')
        return params


