# -*-coding:utf-8 -*-
from collections import defaultdict


def extract_ner(sentence, label):
    entity_dic = defaultdict(list)
    entity = ''
    type = None
    for i,j in zip(sentence, label):
        if 'B-' in j:
            if entity:
                entity_dic[type].append(entity)
            entity = i
            type = j[2:]
        elif 'I-' in j:
            entity +=i
    if entity:
        entity_dic[type].append(entity)
    return entity_dic


def build_entity_dict(data_dir, files, load_data_func):
    sentence_list = []
    label_list = []
    for file in files:
        s,l = load_data_func(data_dir, file)
        sentence_list+=s
        label_list+=l
    entity_dic = defaultdict(list)
    for s, l in zip(sentence_list, label_list):
        dic = extract_ner(s.replace(' ',''), l.split(' '))
        for key, val in dic.items():
            entity_dic[key]+=val

    for key, val in entity_dic.items():
        entity_dic[key] = list(set(val))
    return entity_dic


if __name__ == '__main__':
    import importlib
    import pickle
    ner_dataset = ['cluener', 'people_daily','msra'] # 微博数据太小不考虑了
    mix_dict = defaultdict(list) # combine all extra ner
    for data in ner_dataset:
        data_dir = './data/{}'.format(data)
        module = importlib.import_module('data.{}.preprocess'.format(data))
        if data == 'people_daily':
            # 对我们想要增强的People Daily数据集，我们只用train的样本来构造NER_dict
            files = ['train']
        else:
            files = module.MAPPING
        ner_dict = build_entity_dict(data_dir, files, module.load_data)
        with open('./data/people_daily_augment/{}_ner_dict.pkl'.format(data),'wb' ) as f:
            pickle.dump(ner_dict, f)
        for key, val in ner_dict.items():
            mix_dict[key]+=val

    for key, val in mix_dict.items():
        mix_dict[key] = list(set(val))

    with open('./data/people_daily_augment/extra_ner_dict.pkl', 'wb') as f:
        pickle.dump(mix_dict, f)
