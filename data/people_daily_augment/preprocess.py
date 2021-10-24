# -*-coding:utf-8 -*-
import os
import pickle
from data.base_preprocess import get_instance
from data.people_daily.preprocess import load_data as pl_load_data, TAG2IDX, MAPPING, MAX_SEQ_LEN
from data.tokenizer import Tokenizers

def load_data(data_dir, file_name):
    """
    overwrite MSRA load data with train replacted by train_augment
    All the raw data are still stored in the original directory
    The TFRecord will be stored in the new directory
    """
    data_dir = data_dir.replace('_augment', '')
    if file_name=='train':
        with open(os.path.join(data_dir, 'train_augment.pkl'), 'rb') as f:
            sentences, tags = pickle.load(f)
    else:
        sentences, tags = pl_load_data(data_dir, file_name)
    return sentences, tags


if __name__ == '__main__':
    data_dir = './data/people_daily_augment'

    word_enhance = None
    for tokenizer in Tokenizers:
        for file in MAPPING:
            print('Dumping TF Record for {} word_enhance = {} tokenizer = {}'.\
                  format(file, word_enhance, tokenizer))
            prep = get_instance(tokenizer, MAX_SEQ_LEN, TAG2IDX, MAPPING, word_enhance)
            prep.init_data(data_dir, file, load_data)
            prep.dump_tfrecord()
