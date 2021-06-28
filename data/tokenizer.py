# -*-coding:utf-8 -*-
from bert_base.bert import tokenization
import importlib
import os
import numpy as np

from tools.utils import normalize

TokenizerBert = 'bert'
TokenizerGiga = 'giga'
Tokenizers = [TokenizerGiga, TokenizerBert]


def get_bert_tokenizer(model_dir='./pretrain_model/ch_google/'):
    """
    Bert tokenizer
    Either google_bert or wwm_bert worked, since they share the same word embedding
    """
    bert_tokenizer = tokenization.FullTokenizer(os.path.join(model_dir,
                                                             "vocab.txt"), do_lower_case=True)
    return bert_tokenizer


def get_giga_tokenizer(module='pretrain_model.giga'):
    """
    Giga pretrain character embedding is used in Lattice-LSTM and SoftLexicon
    """
    # giga tokenizer is used in all None bert model
    model = getattr(importlib.import_module(module), 'model') # convert glove to word2vec and return model
    giga_tokenizer = GigaTokenizer(model)

    return giga_tokenizer


class GigaTokenizer(object):
    """
    Fake giga Tokenizer to has same interface as bert tokenizer
    """
    def __init__(self, model):
        self.model = model
        self.vocab2idx = self.get_vocab2idx()

    def get_vocab2idx(self):
        """
        Don't use CLS and SEP. Their random init embedding will impact lstm performance
        """
        vocab2idx = dict([(word, idx) for idx, word in enumerate(self.model.index2word)])
        n_vocab = len(vocab2idx)
        vocab2idx.update({
            '[PAD]': n_vocab,
            '[UNK]': n_vocab+1,
        })
        return vocab2idx

    @property
    def embedding(self):
        embedding = np.array(self.model.vectors)
        addon_embedding = np.random.normal(0, 1, size=(2, self.model.vector_size))
        embedding = np.vstack((embedding, addon_embedding)).astype(np.float32)
        embedding = np.apply_along_axis(normalize, 1 , embedding) # normalize embedding to 1
        return embedding

    @staticmethod
    def full2half(text):
        """
        全角半角转换, giga vocab里面缺少全角字符例如'，'对效果有较大影响，Bert tokenizer没有这个问题
        """
        num = ord(text)
        if num == 0x3000:
            num = 0x20
        elif 0xFF01 <= num <= 0xFF5E:
            num = num - 0xFEE0
        s = chr(num)
        return s

    def tokenize(self, text):
        tokens = []
        for i in text:
            if i.strip():
                i = self.full2half(i)
                if i in self.vocab2idx:
                    tokens.append(i)
                else:
                    tokens.append('[UNK]')
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab2idx[i] for i in tokens]


if __name__ == '__main__':
    tokenizer = get_giga_tokenizer()
    s = '今天天气真好😔'
    tokens = tokenizer.tokenize(s)
    print(tokens )
    tokens +=  ['[PAD]']
    tokenids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokenids)

