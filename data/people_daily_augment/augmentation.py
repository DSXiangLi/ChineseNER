# -*-coding:utf-8 -*-
import pickle
import random
import os
import jieba
import importlib
from tqdm import tqdm

class AugHandler(object):
    def __init__(self, max_sample, change_rate):
        self.max_sample = max_sample
        self.change_rate = change_rate

    def gen_single_sample(self, sentence, label):
        raise NotImplementedError()

    def gen_sample(self, sentence, label, verbose=False):
        new_s = list()
        new_l = list()
        for i in range(self.max_sample):
            s, l = self.gen_single_sample(sentence, label)
            if s!=sentence:
                assert len(s.split(' ')) == len(l.split(' ')), '{}!={}'.format(s,l)
                new_s.append(s)
                new_l.append(l)
                if verbose:
                    print('{} Augment-> {}'.format(sentence, s))
                    print('{} Augment-> {}'.format(label, l))

        return list(new_s), list(new_l)

    @staticmethod
    def chunk_by_tag(sentence, label):
        pre_tag = ''
        s_chunk = []
        l_chunk = []
        for pos, (s,l) in enumerate(zip(sentence.split(' '), label.split(' '))):
            # get NER type
            if l!='O':
                tag = l.split('-')[1]
            else:
                tag = 'O'

            if pre_tag == '':
                pre_tag = tag
                s_chunk = [s]
                l_chunk = [l]

            elif pre_tag == tag and l.split('-')[0]!='B':
                s_chunk.append(s)
                l_chunk.append(l)
            else:
                yield pre_tag, ' '.join(s_chunk), ' '.join(l_chunk)
                pre_tag = tag
                s_chunk=[s]
                l_chunk=[l]
        if s_chunk:
            yield pre_tag, ' '.join(s_chunk), ' '.join(l_chunk)

    @staticmethod
    def chunk_by_word(sentence, label):
        """
        在保护Entity不被改变的情况下返回词粒度
        """
        for tag, s_chunk, l_chunk in AugHandler.chunk_by_tag(sentence, label):
            if tag != 'O':
                yield tag, s_chunk, l_chunk
            else:
                for i, s in enumerate(jieba.cut(''.join(s_chunk.split(' ')))):
                    n = len(s)
                    yield tag, ' '.join(s), ' '.join(['O'] * n)

    @staticmethod
    def chunk_by_sentence(sentence, label):
       sep = {'，'}# 暂时只考虑逗号，其他分隔符对语义影响更大
       s_chunk = []
       l_chunk = []
       for s,l in zip(sentence.split(' '), label.split(' ')):
           s_chunk.append(s)
           l_chunk.append(l)
           if s in sep:
               yield  ' '.join(s_chunk), ' '.join(l_chunk)
               s_chunk = []
               l_chunk = []
       yield ' '.join(s_chunk), ' '.join(l_chunk)


class EntityReplace(AugHandler):
    def __init__(self, ner_dict_file, max_sample, change_rate):
        super(EntityReplace, self).__init__(max_sample, change_rate)
        self.ner_dict_file = ner_dict_file
        self.max_sample = max_sample
        self.ner_dict = self.load_dict()

    def load_dict(self):
        with open(self.ner_dict_file, 'rb') as f:
            ner_dict = pickle.load(f)
        print('Loading NER DICT {}PER {}LOC {}ORG'.format(
            len(ner_dict['PER']), len(ner_dict['LOC']),len(ner_dict['ORG'])
        ))
        return ner_dict

    def select_ner(self, tag, word, label):
        if random.random() < self.change_rate:
            ner = random.choice(self.ner_dict[tag])
            label = ['B-'+tag] + ['I-'+tag] *(len(ner)-1)
            return ' '.join(ner), ' '.join(label)
        else:
            return word, label

    def gen_single_sample(self, sentence, label):
        """
        Sentence, label are split by ' '
        sentence: '今 天 天 气 好'
        lable:'O O O O O'
        return in the same format w/wo ner replaced
        """
        s = []
        l = []
        for tag, s_chunk, l_chunk  in self.chunk_by_tag(sentence, label):
            if tag in self.ner_dict:
                new_s, new_l = self.select_ner(tag, s_chunk, l_chunk)
                s.append(new_s)
                l.append(new_l)
            else:
                s.append(s_chunk)
                l.append(l_chunk)
        s = ' '.join(s)
        l = ' '.join(l)
        return s, l


class SynomReplace(AugHandler):
    """
    对非实体的部分进行随机同义词替换, 这里使用gensim word2vec进行替换
    replace_rate: 控制每个词被替换的概率，再复杂一些可以更高概率替换实体周边词，以及替换概率和句子长度相关
    """
    def __init__(self, model_dir, max_sample, change_rate):
        super(SynomReplace, self).__init__(max_sample, change_rate)
        self.model_dir = model_dir
        self.model= getattr(importlib.import_module(self.model_dir), 'model')

    def select_synom(self, word, label):
        """
        选择Top5相似词，从中任取其一进行替换
        """
        try:
            nn = self.model.most_similar(''.join(word.split(' ')), topn=5)
            if random.random() < self.change_rate:
                word = random.choice(nn)[0]
                label = ' '.join(['O'] * len(word))
                return ' '.join(word), label
            else:
                return word, label
        except:
            return word, label

    def gen_single_sample(self, sentence, label):
        s = []
        l = []
        for tag, s_chunk, l_chunk in self.chunk_by_word(sentence, label):
            if tag =='O':
                new_s, new_l = self.select_synom(s_chunk, l_chunk)
                s.append(new_s)
                l.append(new_l)
            else:
                s.append(s_chunk)
                l.append(l_chunk)
        s = ' '.join(s)
        l = ' '.join(l)
        return s, l


class SentenceShuffle(AugHandler):
    def __init__(self, max_sample, change_rate):
        super(SentenceShuffle, self).__init__(max_sample, change_rate)

    def gen_single_sample(self, sentence, label):
        s = []
        l = []
        pre_s = None
        pre_l = None
        for cur_s, cur_l in self.chunk_by_sentence(sentence, label ):
            if not pre_s:
                pre_s = cur_s
                pre_l = cur_l
                continue
            if random.random() < self.change_rate:
                s.append(cur_s)
                l.append(cur_l)
            else:
                s.append(pre_s)
                l.append(pre_l)
                pre_s = cur_s
                pre_l = cur_l
        s.append(pre_s)
        l.append(pre_l)
        return ' '.join(s), ' '.join(l)


def augment(data_dir, file_name, load_data_func, augment_method_list):
    s_list, l_list = load_data_func(data_dir, file_name)
    counter = 0
    new_s_list = []
    new_l_list = []

    for s, l in tqdm(zip(s_list, l_list), total=len(s_list)):
        for method in augment_method_list:
            new_s, new_l = method(s, l)
            if new_s:
                counter += 1
                new_s_list += new_s
                new_l_list += new_l
    print('NER Augment: Gen {} new samples from {} samples'.format(counter, len(s_list)))

    new_file = os.path.join(data_dir, file_name + '_augment.pkl')
    with open(new_file, 'wb') as f:
        pickle.dump((s_list + new_s_list, l_list + new_l_list), f)
    print('Dump new sample at {}'.format(new_file))


if __name__ == '__main__':
    data_dir = './data/people_daily'
    file_name = 'train'
    from data.people_daily.preprocess import load_data

    ner_handler = EntityReplace('./data/people_daily_augment/extra_ner_dict.pkl', max_sample=3, change_rate=0.2)
    synom_handler = SynomReplace('pretrain_model.ctb50', max_sample=3, change_rate=0.1)
    sent_handler = SentenceShuffle(max_sample=3, change_rate=0.3)
    augment(data_dir, file_name, load_data, [ner_handler.gen_sample,
                                             synom_handler.gen_sample,
                                             sent_handler.gen_sample,
                                             ])
