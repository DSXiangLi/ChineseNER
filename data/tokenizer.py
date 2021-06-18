# -*-coding:utf-8 -*-
from bert_base.bert import tokenization
import importlib
import os
import numpy as np

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
        vocab2idx = dict([(word, idx) for idx, word in enumerate(self.model.index2word)])
        n_vocab = len(vocab2idx)
        vocab2idx.update({
            '[CLS]': n_vocab,
            '[SEP]': n_vocab+1,
            '[PAD]': n_vocab+2,
            '[UNK]': n_vocab+3,
        })
        return vocab2idx

    @property
    def embedding(self):
        embedding = np.array(self.model.vectors)
        addon_embedding = np.random.normal(0, 1, size=(4, self.model.vector_size))
        embedding = np.vstack((embedding, addon_embedding)).astype(np.float32)
        return embedding

    def tokenize(self, text):
        tokens = []
        for i in text:
            if i.strip():
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
    tokens += ['[CLS]'] + ['[SEP]'] + ['[PAD]']
    tokenids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokenids)

