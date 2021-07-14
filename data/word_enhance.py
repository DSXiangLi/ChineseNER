# -*-coding:utf-8 -*-
"""
Word Enhance Method
    supporting : softword, ex_softword, softlexicon
    align method: deal with mismatch between character and bert tokenizer
"""
import os
import jieba
import pickle
import numpy as np
from itertools import chain
from collections import defaultdict
from copy import deepcopy
import importlib
from tools.utils import normalize

# soft2idx, where idx indicate priority in merging
Soft2Idx={
    'S': 0,
    'M': 1,
    'B': 2,
    'E': 3,
    'None': 4,  # used for padding,cls,sep, and None softword
}

# Supported word enhance method
SoftWord = 'softword'
ExSoftWord = 'ex_softword'
SoftLexicon = 'softlexicon'
WordEnhanceMethod = [SoftWord, ExSoftWord, SoftLexicon]

# word enhance constant
MaxWordLen = 10
MaxLexiconLen = 10  # only keep topn words for soft lexicon
NONE_TOKEN= '<None>'
PAD_TOKEN= '<PAD>'
PretrainModel = 'pretrain_model.ctb50' # algin with the lattice and softlexicon

# Lazy Eval Global Variable
model = None
Vocab2IDX = None
VocabFreq = None
N_word = None
VocabEmbedding = None


def model_init():
    # lazy load all model and embedding needed for word enhance
    global model, VocabFreq, Vocab2IDX, VocabEmbedding, N_word
    jieba.initialize() # global initialization
    if model is None:
        print('loading w2v pretrain model {}.It is very very slow, please be patient~~'.format(
            PretrainModel))
        # ctb50/giga are both in glove format, converter is done in __init__
        model = getattr(importlib.import_module(PretrainModel), 'model')
    if Vocab2IDX is None:
        print('Building Vocabulary ...')
        Vocab2IDX = dict([(word, idx) for idx, word in enumerate(model.index2word)])
        VocabFreq = dict([(Vocab2IDX[word], model.wv.vocab[word].count) for word in model.index2word])
        N_word = len(VocabFreq)
        Vocab2IDX.update({
            NONE_TOKEN: N_word,
            PAD_TOKEN: N_word+1
        })
        VocabFreq.update({
            Vocab2IDX[NONE_TOKEN]: 1,
            Vocab2IDX[PAD_TOKEN]: 0
        })
    if VocabEmbedding is None:
        VocabEmbedding = np.vstack((np.array(model.vectors),
                                    np.random.normal(0, 1, size=(model.vector_size)), # None Token
                                    np.zeros(model.vector_size) # Pad Token
                                    )).astype(np.float32)
        VocabEmbedding = np.apply_along_axis(normalize, 1, VocabEmbedding)


def align_with_token(idx_list, tokens, word_enhance):
    """
    Fix mismatch between softword/ex-softword/soft-lexicon and token_ids. Due to
    bert-tokenizer can tokenize few character into 1 token like 1994->19 + ##94
    """
    token_len = [len(i.replace('##','')) if i != '[UNK]' else 1 for i in tokens if i not in ['[CLS]','[SEP]','[PAD]']]
    if len(idx_list) == len(token_len):
        # there is no mismatch between bert tokenizer and character
        return idx_list

    if word_enhance == SoftWord:
        combine_func = combine_softword
    elif word_enhance == ExSoftWord:
        combine_func = combine_ex_softword
    elif word_enhance == SoftLexicon:
        combine_func = combine_soft_lexicon
    else:
        raise ValueError('idx type only supprt {}'.format(','.join(WordEnhanceMethod)))

    pos = 0
    output_list = []
    for tl in token_len:
        if tl == 1:
            output_list.append(idx_list[pos])
        else:
            output_list.append(combine_func(idx_list[pos:pos + tl]))
        pos += tl
    assert len(output_list) == len(token_len)
    return output_list


def combine_softword(idx_list):
    """
    softword: combine multiple char softword id with priority E>B>M>S,
    which is the same order as the Soft2Idx index
    Input: list of softword id
    Output: 1 softword id
    """
    return max(idx_list)


def combine_ex_softword(idx_list):
    """
    Ex-softword: simple add one-hot of multiple char and tail to max(1)
    Input: list of BMES one-hot encoding
    Output: 1 * 4 one-hot encoidng
    """
    one_hot = np.sum(idx_list, axis=0)
    one_hot = [1 if i>0 else 0 for i in one_hot]
    return one_hot


def combine_soft_lexicon(idx_list):
    """
    Soft-Lexicon: combine all lexicon for mutliple char, do truncation and padding
    """
    lexicon_dict = defaultdict(set)
    for lexicon in idx_list:
        for key in Soft2Idx:
            for i in lexicon[key]:
                lexicon_dict[key].add(i)

    return lexicon_dict


def postproc_soft_lexicon(output_list, vocabfreq):
    """
    1.concat B/M/E/S/None into 1 list
    2. truncate/pad soft lexicon to fix length
    3. get normalized weight for each token
    Input
    output_list: list of BMSE soft lexicons{B:[],M:[],E:[],S:[],None}
    vocabfreq: by default use pretrain model vocab freq, otherwise specify data related vocabfreq
    Return
    Ids: seq_len * (4 * MaxLexiconLen)
    Weight: same length as Ids
    """
    seq_ids = []
    seq_weights = []

    def helper(ids):
        n = len(ids)
        if n <= MaxLexiconLen:
            # pad with PAD
            ids = list(ids) + [Vocab2IDX[PAD_TOKEN]] * (MaxLexiconLen -n)
            weight = [vocabfreq.get(i, 1) for i in ids ]
            return ids, weight
        else:
            # truncate to max
            tmp = sorted([(i, vocabfreq.get(i, 1)) for i in ids], key=lambda x: x[1], reverse=True) # sort by frequency
            tmp = tmp[:MaxLexiconLen]
            tmp = list(zip(*tmp))
            return tmp[0], tmp[1]
    # normalize weight for all BMES soft lexicons linked to each token
    for lexicon in output_list:
        ids = []
        weights= []
        total_weight = 0
        for key, val in lexicon.items():
            id, weight = helper(val)
            ids += id
            weights += weight
            total_weight += sum(weight)
        weights = [i/total_weight for i in weights]
        seq_ids.append(ids)
        seq_weights.append(weights)

    return seq_ids, seq_weights


def build_softword(sentence, verbose=False):
    """
    Input: raw sentence
    Output: same length as sentence, using word cut result from jieba
    """
    softword_index = []
    sentence = sentence.replace(' ','') # remove ' ' space in sentence
    words = jieba.cut(sentence)
    for word in words:
        length = len(word)
        if length ==1:
            softword_index.append('S')
        elif length==2:
            softword_index.extend(['B','E'])
        else:
            softword_index.extend(['B'] + (length-2) * ['M'] + ['E'])
    assert len(softword_index)==len(sentence), 'softword len={} != sentence len={}'.format(len(softword_index),
                                                                                           len(sentence))
    if verbose:
        print(sentence)
        print(''.join(softword_index))
    softword_index = [Soft2Idx[i] for i in softword_index]

    return softword_index


def build_ex_softword(sentence, verbose=False):
    """
    Input: raw sentence, a vocabulary to lookup word, verbose whether to print sentence and B/M/E/S
    Output: [one hot encoding of [B, M, E，S]] with length same as sentence
    """
    sentence = sentence.replace(' ', '')
    ex_softword_index = [set() for i in range(len(sentence))]

    for i in range(len(sentence)):
        for j in range(i, min(i+MaxWordLen, len(sentence))):
            word = sentence[i:(j+1)]
            if word in Vocab2IDX:
                if j-i==0:
                    ex_softword_index[i].add('S')
                elif j-i==1:
                    ex_softword_index[i].add('B')
                    ex_softword_index[i+1].add('E')
                else:
                    ex_softword_index[i].add('B')
                    ex_softword_index[j].add('E')
                    for k in range(i+1, j):
                        ex_softword_index[k].add('M')
    if verbose:
        print(sentence)
        print(ex_softword_index)

    # one-hot encoding of B/M/E/S/None/set
    onehot_index = []
    default = [0, 0, 0, 0, 1]
    for index in ex_softword_index:
        if len(index)==0:
            onehot_index.append(default)
        else:
            tmp = [0, 0, 0, 0, 0]
            for i in index:
                tmp[Soft2Idx[i]]=1
            onehot_index.append(tmp)
    return onehot_index


def build_soft_lexicon(sentence, verbose=False):
    """
    Input: raw sentence, a vocabulary to lookup word
    Output: [{'B':[], 'M':[], 'E':[],'S':[]},{'B':[], 'M':[], 'E':[],'S':[]}...]
    """
    sentence = sentence.replace(' ', '')
    default = {'B' : set(), 'M' : set(), 'E' : set(), 'S' :set()}
    soft_lexicon = [deepcopy(default) for i in range(len(sentence))]
    for i in range(len(sentence)):
        for j in range(i, min(i+MaxWordLen, len(sentence))):
            word = sentence[i:(j + 1)]
            if word in Vocab2IDX:
                if j-i==0:
                    soft_lexicon[i]['S'].add(word)
                elif j-i==1:
                    soft_lexicon[i]['B'].add(word)
                    soft_lexicon[i+1]['E'].add(word)
                else:
                    soft_lexicon[i]['B'].add(word)
                    soft_lexicon[j]['E'].add(word)
                    for k in range(i+1, j):
                        soft_lexicon[k]['M'].add(word)
        for key, val in soft_lexicon[i].items():
            if not val:
                # eg.no matching E soft lexicon, fill in with None Token
                soft_lexicon[i][key].add(NONE_TOKEN)

    if verbose:
        print(sentence)
        print(soft_lexicon)

    for lexicon in soft_lexicon:
        for key,val in lexicon.items():
            lexicon[key] = [Vocab2IDX[i] for i in val]

    return soft_lexicon


def prebuild_weight(data_dir, sentences):
    """
    Pretrain static word count used for soft lexicon
    """
    print('Pre build soft lexicon weight for {}'.format(data_dir))
    lexicon_counter = defaultdict(int)
    for i in sentences:
        soft_lexicon = build_soft_lexicon(i)
        for item in soft_lexicon:
            for val in chain(*item.values()):
                lexicon_counter[val] +=1
    file_path = os.path.join(data_dir, 'lexicon_weight.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(lexicon_counter, f)
    print('weight is dump at {}'.format(file_path))


if __name__ == '__main__':
    # pre build word frequency used by msra, people_daily
    test = False
    if test:
        model_init()
        from bert_base.bert import tokenization
        tokenizer = tokenization.FullTokenizer("./pretrain_model/ch_google/vocab.txt", do_lower_case=True)
        ## Test make feature
        s1 ='1994年海钓比赛地点在厦门与金门之间的海域'
        s2 ='这座依山傍水的博物馆由国内一流的设计师主持设计'
        # Test build softword
        r1 = build_softword(s1, True)
        r2 = build_softword(s2, True)

        # Test build ExSoftword
        r1 = build_ex_softword(s1, True)
        r2 = build_ex_softword(s2, True)

        # Test SoftLexicon
        s1 = '为 了 跟 踪 '
        r1 = build_soft_lexicon(s1, True)
        r2 = build_soft_lexicon(s2, True)

        ## Test align with bert tokenizer
        r1 = align_with_token( build_softword(s1, True), tokenizer.tokenize(s1), 'softword')
        r1 = align_with_token( build_ex_softword(s1, True), tokenizer.tokenize(s1), 'ex_softword')
        r1 = align_with_token( build_soft_lexicon(s1, True), tokenizer.tokenize(s1), 'soft_lexicon')

        # test post proc of soft-lexicon
        r1 = build_soft_lexicon(s1, True)
        r2 = align_with_token(r1, tokenizer.tokenize(s1), 'soft_lexicon')
        ids, weight = postproc_soft_lexicon(r2)