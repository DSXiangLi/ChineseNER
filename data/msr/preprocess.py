# -*-coding:utf-8 -*-

from data.base_preprocess import get_instance, read_text
from itertools import chain
from data.tokenizer import Tokenizers

TAG2IDX = {
    '[PAD]': 0,
    'B': 1,
    'I': 2,
    'E': 3,
    'S': 4,
    '[CLS]': 5,
    '[SEP]': 6
}

MAPPING = {
            'training': 'train',
            'test_gold': 'valid',
            'test': 'predict'
}

MAX_SEQ_LEN = 150


def gen_tag(length):
    if length == 1:
        return 'S'
    elif length == 2:
        return 'B E'
    else:
        return ' '.join(['B'] + ['I'] * (length - 2) + ['E'])


def load_data(data_dir, file_name):
    """
    Load line, word are separated by ' ', use len of tokenizer to avoid mismatch with bert tokenizer
    """
    lines = read_text(data_dir, 'msr_{}.utf8'.format(file_name))
    sentences = []
    tags = []
    for line in lines:
        if line == '':
            continue
        else:
            tag = [gen_tag(len(token)) for token in line.split(' ') if
                   token not in ['', '"']]
            # split word into char to avoid bert tokenizer from merging tokens
            sentence = chain(*[ [i for i in token ] for token in line.split(' ') if token not in ['', '"']])

        tags.append(' '.join(tag))
        sentences.append(' '.join(sentence))
    return sentences, tags


if __name__ == '__main__':
    data_dir = './data/msr'
    # for CWS + NER multi-task, only word_enhance=None is supported
    word_enhance = None
    for tokenizer in Tokenizers:
        for file in MAPPING:
            print('Dumping TF Record for {} word_enhance = {} tokenizer = {}'. \
                  format(file, word_enhance, tokenizer))
            prep = get_instance(data_dir, file, tokenizer, MAX_SEQ_LEN, TAG2IDX, MAPPING,
                                load_data, word_enhance)
            prep.dump_tfrecord()