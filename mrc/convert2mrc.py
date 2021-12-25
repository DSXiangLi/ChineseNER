# -*-coding:utf-8 -*-
"""
把数据格式转化成MRC所需的数据格式, 并dump在原folder内
{
    'title': '北京天气真好',
    'label': [
        {'span':'北京',
        'tag':'LOC',
        'start_pos':0,
        'end_pos':1
        }
    ]
}
"""
import importlib
import os
import json


def convert2mrc(sentence, tag_str):
    """
    Transform Sentence and tag to MRC style
    """
    sample = {}
    sample['title'] = sentence
    sample['label'] = []
    start_pos = None
    end_pos = None
    span = ''
    tag = ''
    for pos, (s, l) in enumerate(zip(sentence.split(), tag_str.split())):
        if 'B-' in l :
            if span:
                sample['label'].append({
                    'span': span,
                    'tag': tag,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                })
            span = s
            start_pos = pos
            end_pos = pos
            tag= l.split('-')[1]

        elif 'I-' in l:
            end_pos = pos
            span += s
        else:
            if span:
                sample['label'].append({
                    'span': span,
                    'tag': tag,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                })
                span = ''
                tag= ''
    if span:
        sample['label'].append({
            'span': span,
            'tag': tag,
            'start_pos': start_pos,
            'end_pos': end_pos,
        })
    return sample


def run(data_name):
    data_dir = './data/{}'.format(data_name)
    module = importlib.import_module('data.{}.preprocess'.format(data_name))

    print('Coverting {} to MRC format'.format(data_name))
    for file_name, rename in module.MAPPING.items():
        sentences, tags = module.load_data(data_dir, file_name)
        samples = []
        for sentence, tag in zip(sentences, tags):
            sample = convert2mrc(sentence, tag)
            samples.append(sample)

        with open(os.path.join(data_dir, module.MAPPING[file_name]+'_mrc.txt'), 'w') as f:
            for i in samples:
                f.write(json.dumps(i,ensure_ascii=False) + '\n')


if __name__ == '__main__':
    data_list = ['msra', 'people_daily']
    for data in data_list:
        run(data)
