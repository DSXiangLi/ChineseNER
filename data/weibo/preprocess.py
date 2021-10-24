# -*-coding:utf-8 -*-
from data.base_preprocess import read_text
import codecs as cs
import os

TAG2IDX = {
    '[PAD]': 0,
    'O': 1,
    'B-ORG': 2,
    'I-ORG': 3,
    'B-PER': 4,
    'I-PER': 5,
    'B-LOC': 6,
    'I-LOC': 7,
    '[CLS]': 8,
    '[SEP]': 9
}

MAX_SEQ_LEN = 150

MAPPING = {
    'train': 'train',
    'dev': 'valid',
    'test': 'predict'
}


def load_data(data_dir, file_name):
    """
    Load line and generate sentence and tag list with char split by ' '
    sentences: ['今 天 天 气 好 ', ]
    tags: ['O O O O O ', ]
    """
    sentences = []
    tags = []
    tag = []
    sentence = []
    with cs.open(
    os.path.join(data_dir, 'weiboNER_2nd_conll.{}'.format(file_name)),
    'r', encoding='utf-8') as f:
        tokens = f.readlines()
        for i in tokens:
            i = i.strip()
            if i == '':
                # Here join by ' ' to avoid bert_tokenizer merging tokens
                sentences.append(' '.join(sentence))
                tags.append(' '.join(tag))
                tag = []
                sentence = []
            else:
                s, t = i.split('\t')
                tag.append(t)
                sentence.append(s[:-1])

    return sentences, tags


if __name__ == '__main__':
    data_dir = './data/weibo'
    file_name = 'dev'
    sentence_list, label_list = load_data(data_dir, file_name)