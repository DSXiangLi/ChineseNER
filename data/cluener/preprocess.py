# -*-coding:utf-8 -*-
import os
from data.base_preprocess import read_text
import json

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


def gen_label(label, label_type, pos_list):
    for pos in pos_list:
        for pair in pos:
            start = pair[0]
            end = pair[1]
            label[start] ='B-' + label_type
            label[(start+1):(end+1)] = ['I-'+ label_type] * (end-start)
    return label


def load_data(data_dir, file_name):
    """
    这里我们只用cluener来构建额外实体字典，不用来训练模型
    为了和其他数据机保持一致这里我们把细分实体映射会PER，LOC，ORG

    """
    samples = read_text(data_dir, file_name+'.json')
    sentence_list = []
    label_list = []
    label_type = {
        'name': 'PER',
        'company': 'ORG',
        'goverment': 'ORG',
        'organization': 'ORG',
        'address': 'LOC',
        'scene': 'LOC'
    }
    for s in samples:
        s = json.loads(s)
        sentence = s['text']
        sentence_list.append(sentence)
        label = ['I'] * len(sentence)
        for key, val in s['label'].items():
            if key in label_type:
                label = gen_label(label, label_type[key], val.values())
        label_list.append(' '.join(label))

    return sentence_list, label_list

MAPPING = {
    'train': 'train'
}

if __name__ == '__main__':
    data_dir = './data/cluener'
    file_name = 'train'
    sentence_list, label_list = load_data(data_dir, file_name)