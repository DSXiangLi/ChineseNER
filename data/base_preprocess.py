# -*-coding:utf-8 -*-
import os
import pickle
import re
import tensorflow as tf
from itertools import chain
from functools import partial
import importlib
from data.word_enhance import (align_with_token, build_softword, build_ex_softword, build_soft_lexicon,
                               WordEnhanceMethod, SoftWord, SoftLexicon, ExSoftWord, BiChar, build_bichar,
                               Soft2Idx, MaxLexiconLen, bigiga50_handler, ctb50_handler,
                               postproc_soft_lexicon, prebuild_weight)
from data.tokenizer import TokenizerGiga, TokenizerBert


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_prefix_surfix(model_name):
    """
    Infer word_enhance and tokenizer type from model name
    1. model_name start with bert use bert tokenizer
    2. model_name end with word enhance use word enhance
    """
    word_enhance = re.search('({})|({})|({})|({})'.format(SoftWord, SoftLexicon, ExSoftWord, BiChar), model_name)
    word_enhance = word_enhance.group() if word_enhance else None

    tokenizer_type = re.search('({})'.format(TokenizerBert), model_name)
    tokenizer_type = tokenizer_type.group() if tokenizer_type else TokenizerGiga
    return word_enhance, tokenizer_type


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
        #len(Soft2Idx)=5, including B/M/E/S/None
        feature_proto.update({
            'ex_softword_ids': tf.io.FixedLenFeature([max_seq_len*len(Soft2Idx)], dtype=tf.int64)
        })
    elif word_enhance == BiChar:
        feature_proto.update({
            'bichar_ids': tf.io.FixedLenFeature([max_seq_len], dtype=tf.int64)
        })
    else:
        #rm None from Soft2Idx
        feature_proto.update({
            'softlexicon_ids': tf.io.FixedLenFeature([max_seq_len*(len(Soft2Idx)-1)*MaxLexiconLen], dtype=tf.int64),
            'softlexicon_weights': tf.io.FixedLenFeature([max_seq_len*(len(Soft2Idx)-1)*MaxLexiconLen], dtype=tf.float32)
        })

    return feature_proto


def get_instance(tokenizer_type, max_seq_len, tag2idx, mapping=None, word_enhance=None, **kwargs):
    # Init instance with different cls, and dynamic add data specific load_data func to instance
    assert word_enhance in [None]+WordEnhanceMethod, 'word_enhance must in {}'.format(','.join(WordEnhanceMethod))

    if word_enhance is None:
        cls = BasicProc
    else:
        if word_enhance == SoftWord:
            cls = SoftwordProc
        elif word_enhance == ExSoftWord:
            cls = ExSoftwordProc
        elif word_enhance == BiChar:
            cls = BicharProc
        else:
            cls = SoftLexiconProc

    instance = cls(tokenizer_type, max_seq_len, tag2idx, mapping, **kwargs)
    return instance


def read_text(data_dir, filename):
    data = []
    with open(os.path.join(data_dir, filename), 'r') as f:
        for line in f:
            data.append(line.strip())
    return data


def tf_string_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(i, encoding='UTF-8') for i in value]))


def tf_int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tf_float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


FeatureSchema = {
    'tokens': tf_string_feature,
    'token_ids': tf_int_feature,
    'mask': tf_int_feature,
    'segment_ids': tf_int_feature,
    'labels': tf_string_feature,
    'label_ids': tf_int_feature,
    'seq_len': tf_int_feature,
    'softword_ids': tf_int_feature,
    'bichar_ids': tf_int_feature,
    'ex_softword_ids': tf_int_feature,
    'softlexicon_ids': tf_int_feature,
    'softlexicon_weights': tf_float_feature,
    'task_ids': tf_int_feature, # for multitask only
}


class BasicProc(object):
    def __init__(self, tokenizer_type, max_seq_len, tag2idx, mapping):
        self.max_seq_len = max_seq_len
        self.tag2idx = tag2idx
        # get certain tokenizer
        self.tokenizer = getattr(importlib.import_module('data.tokenizer'), 'get_{}_tokenizer'.format(tokenizer_type))()
        self.mapping = mapping
        self.word_enhance = None # used to distinguish basic tfrecord from other word_enhance tfrecord
        self.tokenizer_type = tokenizer_type # used to distinguish tfrecord for bert model from other None bert model
        # data specific attr, not needed in inference
        self.data_dir = None
        self.file_name = None
        self.sentence_list = None
        self.tag_list = None

    def init_data(self, data_dir, file_name, load_data_func):
        self.data_dir = data_dir
        self.file_name = file_name
        self.sentence_list, self.tag_list = load_data_func(data_dir, file_name)

    @property
    def rename_file(self):
        return self.mapping[self.file_name]

    def format_sequence(self, seq):
        """
        For Bert model: add [CLS] at beginning, [SEP] in the end, then do padding
        For non-bert model: only do padding
        """
        if self.tokenizer_type == TokenizerBert:
            seq = seq[:(self.max_seq_len-2)]
            seq = ['[CLS]'] + seq + ['[SEP]']
        else:
            seq = seq[: self.max_seq_len]
        seq_len = len(seq)

        seq += ['[PAD]'] * (self.max_seq_len - seq_len)
        return seq, seq_len

    def build_seq_feature(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens, seq_len = self.format_sequence(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * self.max_seq_len
        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        assert len(tokens) == self.max_seq_len
        assert len(token_ids) == self.max_seq_len
        assert len(mask) == self.max_seq_len
        return DotDict({'tokens': tokens, 'token_ids': token_ids,
                        'segment_ids': segment_ids, 'mask': mask, 'seq_len': seq_len})

    def build_tag_feature(self, tag):
        tags = tag.split(' ')
        labels, label_len = self.format_sequence(tags)
        label_ids = [self.tag2idx[i] for i in labels]
        assert len(labels) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len
        return DotDict({'labels': labels, 'label_ids': label_ids, 'label_len': label_len})

    def build_feature(self, sentence, tag):
        f_seq = self.build_seq_feature(sentence)
        f_label = self.build_tag_feature(tag)
        assert f_seq.seq_len == f_label.label_len,\
            'sentence = {}... {}!={} n_token!=n_tag'.format(sentence[:10], f_seq.seq_len, f_label.label_len)

        return DotDict({**f_seq, **f_label})

    def build_tf_feature(self, feature):
        tf_feature = {}
        for key, val in feature.items():
            if key in FeatureSchema:
                tf_feature[key] = FeatureSchema[key](val)
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
        if self.tokenizer_type == TokenizerGiga:
            params.update({
                'embedding': self.tokenizer.embedding # add pretrain token embedding
            })
        return params

    def dump_tfrecord(self):
        n_sample = 0
        n_invalid_sample = 0
        with tf.io.TFRecordWriter(os.path.join(self.data_dir, '_'.join(filter(None, [self.tokenizer_type,
                                                                                     self.rename_file,
                                                                                     self.word_enhance])) + '.tfrecord')) as writer:
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
            with open(os.path.join(self.data_dir, '_'.join(filter(None, [self.tokenizer_type,
                                                                         self.word_enhance,
                                                                         'data_params.pkl']))), 'wb') as f:
                pickle.dump(params, f)

####################################
# Following are word enhance class #
####################################


class BicharProc(BasicProc):
    def __init__(self, tokenizer_type, max_seq_len, tag2idx, mapping):
        super(BicharProc, self).__init__(tokenizer_type, max_seq_len, tag2idx, mapping)
        self.word_enhance = BiChar
        bigiga50_handler.init()

    def format_soft_seq(self, seq):
        default_label_encoding = bigiga50_handler.vocab2idx[bigiga50_handler.pad_token]
        if self.tokenizer_type == TokenizerBert:
            seq = seq[:(self.max_seq_len - 2)]
            seq = [default_label_encoding] + seq + [default_label_encoding]
        else:
            seq = seq[:self.max_seq_len]
        len_seq = len(seq)

        seq += [default_label_encoding] * (self.max_seq_len - len_seq)
        return seq

    def build_seq_feature(self, sentence):
        """
        Add additional softword ids into sequence feature
        """
        f_seq = super().build_seq_feature(sentence)
        bichar_ids = build_bichar(sentence, False)
        if self.tokenizer_type == TokenizerBert:
            # for character tokenizer, no need to align with tokenizer
            bichar_ids = align_with_token(bichar_ids, f_seq.tokens, word_enhance=self.word_enhance)
        bichar_ids = self.format_soft_seq(bichar_ids)
        f_seq.bichar_ids = bichar_ids
        return f_seq

    def build_data_params(self, n_sample):
        params = super().build_data_params(n_sample)
        params['bichar_embedding'] = bigiga50_handler.vocab_embedding
        params['word_enhance_dim'] = bigiga50_handler.embedding_dim
        params['vocab2idx'] = bigiga50_handler.vocab2idx
        return params


class SoftwordProc(BasicProc):
    def __init__(self, tokenizer_type, max_seq_len, tag2idx, mapping):
        super(SoftwordProc, self).__init__(tokenizer_type, max_seq_len, tag2idx, mapping)
        self.word_enhance = SoftWord
        self.soft2idx = Soft2Idx

    def format_soft_seq(self, seq):
        default_label_encoding = self.soft2idx['None']
        if self.tokenizer_type == TokenizerBert:
            seq = seq[:(self.max_seq_len - 2)]
            seq = [default_label_encoding] + seq + [default_label_encoding]
        else:
            seq = seq[:self.max_seq_len]
        len_seq = len(seq)

        seq += [default_label_encoding] * (self.max_seq_len - len_seq)
        return seq

    def build_seq_feature(self, sentence):
        """
        Add additional softword ids into sequence feature
        """
        f_seq = super().build_seq_feature(sentence)
        softword_ids = build_softword(sentence, False)
        if self.tokenizer_type == TokenizerBert:
            # for character tokenizer, no need to align with tokenizer
            softword_ids = align_with_token(softword_ids, f_seq.tokens, word_enhance=self.word_enhance)
        softword_ids = self.format_soft_seq(softword_ids)
        f_seq.softword_ids = softword_ids
        return f_seq

    def build_data_params(self, n_sample):
        params = super().build_data_params(n_sample)
        params['soft2idx'] = self.soft2idx
        params['word_enhance_dim'] = len(self.soft2idx) # number of word enhance label encoder
        return params


class ExSoftwordProc(BasicProc):
    def __init__(self, tokenizer_type, max_seq_len, tag2idx, mapping):
        super(ExSoftwordProc, self).__init__(tokenizer_type, max_seq_len, tag2idx, mapping)
        self.word_enhance = ExSoftWord
        self.soft2idx = Soft2Idx

    def format_soft_seq(self, seq):
        default_one_hot = [0] * len(self.soft2idx)
        default_one_hot[self.soft2idx['None']] = 1 #[0,0,0,0,1] for cls,sep,pad

        if self.tokenizer_type == TokenizerBert:
            seq = seq[:(self.max_seq_len - 2)]
            seq = [default_one_hot] + seq + [default_one_hot]
        else:
            seq = seq[:self.max_seq_len]
        len_seq = len(seq)

        seq += [default_one_hot] * (self.max_seq_len - len_seq)
        # flatten to 1 dimension for tf input
        seq = list(chain(*seq))
        return seq

    def build_seq_feature(self, sentence):
        f_seq = super().build_seq_feature(sentence)
        ex_softword_ids = build_ex_softword(sentence, False)
        if self.tokenizer_type == TokenizerBert:
            # for character tokenizer, no need to align with tokenizer
            ex_softword_ids = align_with_token(ex_softword_ids, f_seq.tokens, word_enhance=self.word_enhance)
        f_seq.ex_softword_ids = self.format_soft_seq(ex_softword_ids)
        return f_seq

    def build_data_params(self, n_sample):
        params = super().build_data_params(n_sample)
        params['soft2idx'] = self.soft2idx
        params['word_enhance_dim'] = len(self.soft2idx) # dimension of one-hot word enhance vector
        return params


class SoftLexiconProc(BasicProc):
    def __init__(self, tokenizer_type, max_seq_len, tag2idx, mapping, default_weight=True):
        super(SoftLexiconProc, self).__init__(tokenizer_type, max_seq_len, tag2idx, mapping)
        self.word_enhance = SoftLexicon # surfix = word enhance method
        self.soft2idx=Soft2Idx
        ctb50_handler.init()
        self.default_weight = default_weight # True will use pretrain model vocabFreq, otherwise calculate by input data
        self.postproc_func = self.init_weight()

    def init_weight(self):
        if self.default_weight:
            print('Using default vocab frequency from pretrain model')
            return partial(postproc_soft_lexicon, vocabfreq=ctb50_handler.vocab_freq)
        # calculate weight for softlexion using training data
        if self.rename_file == 'train':
            prebuild_weight(self.data_dir, self.sentence_list)
        # load vocab frequency
        with open(os.path.join(self.data_dir,'lexicon_weight.pkl'), 'rb') as f:
            self.vocabfreq = pickle.load(f)
        return partial(postproc_soft_lexicon, vocabfreq=self.vocabfreq)

    def format_soft_seq(self, seq, type='ids'):
        if type == 'weight':
            default_encoding = [0.0] * len(seq[0])
        else:
            default_encoding = [0] * len(seq[0])

        if self.tokenizer_type == TokenizerBert:
            seq = seq[:(self.max_seq_len-2)]
            seq = [default_encoding] + seq + [default_encoding]
        else:
            seq = seq[:self.max_seq_len]
        len_seq = len(seq)

        seq += [default_encoding] * (self.max_seq_len - len_seq)
        seq = list(chain(*seq)) # flatten to 1 dimension
        return seq

    def build_seq_feature(self, sentence):
        """
        getting all possible soft lexicon, truncate/pad to max len and calculate normalized weight
        softlexicon_ids: max_seq_len * (4 * MaxLexiconLen)
        softlexicon_weight: same shape as ids
        """
        f_seq = super().build_seq_feature(sentence)
        soft_lexicon = build_soft_lexicon(sentence, False)
        if self.tokenizer_type == TokenizerBert:
            # for character tokenizer, no need to align with tokenizer
            soft_lexicon = align_with_token(soft_lexicon, f_seq.tokens, word_enhance=self.word_enhance)
        ids, weights = self.postproc_func(soft_lexicon)
        f_seq.softlexicon_ids = self.format_soft_seq(ids)
        f_seq.softlexicon_weights = self.format_soft_seq(weights, type='weight')
        return f_seq

    def build_data_params(self, n_sample):
        # pass in pretrain model vocab Embedding
        params = super().build_data_params(n_sample)
        params['soft2idx'] = self.soft2idx
        params['word_enhance_dim'] = len(self.soft2idx)-1 # 4=B/M/E/S
        params['max_lexicon_len'] = MaxLexiconLen
        params['vocab2idx'] = ctb50_handler.vocab2idx
        params['word_embedding'] = ctb50_handler.vocab_embedding
        return params

