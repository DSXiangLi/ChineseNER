# -*-coding:utf-8 -*-
import os
from data.base_preprocess import get_instance, read_text
from data.word_enhance import WordEnhanceMethod
from data.tokenizer import Tokenizers

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

MAPPING = {'train': 'train',
            'val': 'valid',
            'test': 'predict'
           }


def load_data(data_dir, file_name):
    """
    Load sentence and tags
    """
    sentences = read_text(data_dir, os.path.join(file_name, 'sentences.txt'))
    tags = read_text(data_dir, os.path.join(file_name, 'tags.txt'))
    assert len(sentences) == len(tags)
    return sentences, tags


if __name__ == '__main__':
    data_dir = './data/msra'

    for tokenizer in Tokenizers:
         for word_enhance in [None]+WordEnhanceMethod:
            for file in MAPPING:
                print('Dumping TF Record for {} word_enhance = {} tokenizer = {}'.\
                      format(file, word_enhance, tokenizer))
                prep = get_instance(data_dir, file, tokenizer, MAX_SEQ_LEN, TAG2IDX, MAPPING,
                                    load_data, word_enhance)
                prep.dump_tfrecord()

