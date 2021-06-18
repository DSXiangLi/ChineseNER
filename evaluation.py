# -*-coding:utf-8 -*-
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from itertools import chain
from seqeval.metrics import classification_report as ner_cls_report
from sklearn.metrics import classification_report as tag_cls_report
from collections import defaultdict
from tools.predict_utils import process_prediction

from data.base_preprocess import extract_prefix_surfix


class SingleEval(object):
    def __init__(self, model_name, data, verbose=True):
        self.surfix, self.prefix = extract_prefix_surfix(model_name)
        self.model_name = model_name
        self.data = data
        self.prediction = None
        self.idx2tag = None
        self.label_entity = None
        self.pred_entity = None
        self.sentences = None
        self.verbose = verbose
        self.init()

    def init(self):
        with open('./data/{}/{}_predict.pkl'.format(self.data, self.model_name), 'rb') as f:
            prediction = pickle.load(f)
        with open('./data/{}/{}'.format(self.data,
                                        '_'.join(filter(None, [self.prefix, self.surfix, 'data_params.pkl']))), 'rb') as f:
            data_params = pickle.load(f)
        self.idx2tag = data_params['idx2tag']
        self.prediction = [process_prediction(i, self.idx2tag) for i in prediction]

    def tag_eval(self):
        y_true = list(chain(*[i['label_ids'] for i in self.prediction]))
        y_pred = list(chain(*[i['pred_ids'] for i in self.prediction]))
        target_names = [i for i in self.idx2tag.values() if i not in ['[PAD]', '[CLS]', '[SEP]']]
        report = tag_cls_report(y_true, y_pred, target_names=target_names, output_dict=True, digits=3)
        if self.verbose:
            print('Tag Level Evaluation')
            print(tag_cls_report(y_true, y_pred, target_names=target_names))
        return report

    def entity_eval(self):
        y_true = [i['labels'] for i in self.prediction]
        y_pred = [i['preds'] for i in self.prediction]
        report = ner_cls_report(y_true, y_pred, output_dict=True, digits=3)
        if self.verbose:
            print('Entity Level Evaluation Strict')
            print(ner_cls_report(y_true, y_pred))
        return report

    def sample_topn(self, topn):
        samples = np.random.choice(range(len(self.prediction)), size=topn, replace=False)
        for i in samples:
            print('sentence = {}'.format(self.prediction[i]['sentence']))
            print('Entity = {}'.format(self.prediction[i]['label_entity']))
            print('Pred = {}'.format(self.prediction[i]['pred_entity']))

    def gen_report(self):
        if self.verbose:
            print('Data={} Model={} Evaluation {} total sample'.format(self.data,
                                                                       self.model_name,
                                                                      len(self.prediction)))
        tag_report = self.tag_eval()
        entity_report = self.entity_eval()
        return tag_report, entity_report


class MultiEval(object):
    def __init__(self, model_name_list, data):
        self.data = data
        self.model_name_list = model_name_list
        self.single_eval_list = [SingleEval(i, data, verbose=False) for i in model_name_list]

    def gen_report(self):
        """
        Get Entity Level Macro Average and Tag Level Macro Average
        """
        print('Data={} Evaluation'.format(self.data))
        self.metrics = defaultdict(dict)
        for model, se in zip(self.model_name_list, self.single_eval_list):
            tr, er = se.gen_report()
            self.metrics['tag'].update({
                model:  tr['weighted avg']
            })
            self.metrics['entity'].update({
                model:  er['weighted avg']
            })

        self.pprint(self.metrics)

    @staticmethod
    def pprint(metrics):
        format_dict = {
            'precision': '{:.3%}',
            'recall': '{:.3%}',
            'f1-score': '{:.3%}',
            'support': '{:.0f}'
        }
        for level in ['entity', 'tag']:
            print('='*10+' {} level Evaluation '.format(level)+'='*10)
            df = pd.DataFrame(metrics[level]).transpose()
            df.sort_values(by='f1-score', ascending=False, inplace=True)
            for i in df.columns:
                df[i] = df[i].map(format_dict[i].format)
            print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='if multiple model split by ,',
                        required=True)
    parser.add_argument('--data', type=str, help='which data to evaluate on ',
                        required=False, default='msra')
    parser.add_argument('--topn', type=int, help='Whether to clear existing model',
                        required=False, default=10)
    args = parser.parse_args()
    model_name_list = args.model_name.split(',')
    if len(model_name_list) ==1:
        # If only 1 model is provided, Single Evaluate
        se = SingleEval(args.model_name, args.data)
        _ = se.gen_report()
        se.sample_topn(args.topn)
    else:
        # Otherwise compare F1 for multi model
        me = MultiEval(model_name_list, args.data)
        me.gen_report()
