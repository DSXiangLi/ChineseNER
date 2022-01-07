# -*-coding:utf-8 -*-
import os
import argparse
import pickle
import json
import tensorflow as tf

from config import RUN_CONFIG
from mrc.dataset import MRCBIODataset, Tag2Idx, Idx2Tag
from mrc.model import build_model_fn
from mrc.evaluation import bio_extract_entity
from tools.utils import clear_model, build_estimator
from tools.logger import getLogger
from seqeval.metrics import classification_report as ner_cls_report
from tools.loss import LossHP, LossFunc
import numpy as np

TP = {
    'dtype': tf.float32,
    'lr': 5e-6,
    'log_steps': 100,
    'pretrain_dir': './pretrain_model/ch_google',  # pretrain Bert-Model
    'batch_size': 32,
    'epoch_size': 40,
    'embedding_dropout': 0.3,
    'warmup_ratio': 0.1,
    'early_stop_ratio': 1,  # stop after ratio * steps_per_epoch,
    'max_seq_len': 170,
    'label_size': 2,
    'diff_lr_times': {'logit': 100}
}

model_name = 'MRC'
EXPORT_DIR = './serving_model/{}'
CKPT_DIR = './checkpoint/ner_{}_{}'
DATA_DIR = './data/{}'


def main(args):
    model_dir = CKPT_DIR.format('msra', model_name)
    data_dir = DATA_DIR.format(args.data)

    # get loss function given args
    loss_hp = loss_hp_parser.parse(args)
    TP['loss_func'] = LossFunc[loss_name](**loss_hp)

    if args.clear_model:
        clear_model(model_dir)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    input_pipe = MRCBIODataset(data_dir, TP['batch_size'], TP['max_seq_len'], Tag2Idx)
    input_pipe.build_feature('train')

    TP.update({
        'label_size': input_pipe.label_size,
        'n_samples': input_pipe.n_samples,
        'steps_per_epoch': input_pipe.steps_per_epoch,
        'num_train_steps': int(input_pipe.steps_per_epoch * TP['epoch_size'])
    })

    estimator = build_estimator(TP, model_dir, build_model_fn(), args.use_gpu, RUN_CONFIG)

    logger = getLogger('train', log_dir=model_dir) # init file handler in checkpoint dir
    logger.info('=' * 10 + 'Train Parameters' + '=' * 10)
    logger.info(TP)

    if args.do_train:
        # Run Train & Evaluate
        logger.info('=' * 10 + 'Train {} for {}'.format(model_name, args.data) + '=' * 10)
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator, metric_name='loss',
            max_steps_without_decrease=int(TP['steps_per_epoch'] * TP['early_stop_ratio'])
        )
        train_spec = tf.estimator.TrainSpec(input_pipe.build_input_fn(),
                                            hooks=[early_stopping_hook], max_steps=TP['num_train_steps'])
        input_pipe.build_feature('valid')
        eval_spec = tf.estimator.EvalSpec(input_pipe.build_input_fn(is_predict=True),
                                          steps=TP['steps_per_epoch'], throttle_secs=60)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.do_export:
        # Export Model for serving
        logger.info('Exporting Model for serving  at {}'.format(EXPORT_DIR.format(model_name)))
        estimator._export_to_tpu = False
        estimator.export_saved_model(EXPORT_DIR.format(model_name), lambda: input_pipe.build_serving_proto())

    if args.do_eval:
        # Do prediction when train finished
        logger.info('Run Evaluation')
        input_pipe.build_feature('predict')
        predictions = estimator.predict(input_fn=input_pipe.build_input_fn(is_predict=True))
        predictions = [i for i in predictions]

        logger.info('Dumping prediction at {}'.format('./data/{}/{}_predict.pkl'.format(args.data, model_name)))
        with open('{}/{}_predict.pkl'.format(data_dir, model_name), 'wb') as f:
            pickle.dump(predictions, f)

        ## Extract Entity
        logger.info('Extracting Entity. Dumping at {}'.format('{}/{}_entity_predict.txt'.format(data_dir, model_name)))
        result = []
        for pred, sample, in zip(predictions, input_pipe.samples):
            pred_ids = np.argmax(pred['probs'], axis=-1) * pred['text_mask']
            pred_entity_list = bio_extract_entity(pred_ids, sample['input'], Tag2Idx)
            label_entity_list = bio_extract_entity(sample['label_ids'], sample['input'], Tag2Idx)
            result.append({'text': ''.join(sample['text']),
                            'tag': sample['tag'],
                            'pred_entity_list': pred_entity_list,
                            'true_entity_list': label_entity_list})

        with open('{}/{}_entity_predict.txt'.format(data_dir, model_name), 'w') as f:
            for i in result:
                f.write(json.dumps(i, ensure_ascii=False)+'\n')

        ## Span Evaluation
        logger.info('Running Span Level Evaluation. Writting to train.log in checkpoint')
        y_true = []
        y_pred = []
        for pred, sample in zip(predictions, input_pipe.samples):
            tag = sample['tag']
            pred_ids = np.argmax(pred['probs'], axis=-1) * pred['text_mask']
            pred_ids = np.multiply(pred_ids, pred['text_mask'])
            y_true.append(['O' if i==Tag2Idx['O'] else '-'.join((Idx2Tag[i],tag))  for i in sample['label_ids']])
            y_pred.append(['O' if i==Tag2Idx['O'] else '-'.join((Idx2Tag[i],tag))  for i in pred_ids])

        eval_report = ner_cls_report(y_true, y_pred)
        logger.info(eval_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='people_daily', help='which data to use[msra, people_daily]' )
    parser.add_argument("--loss", default='ce', type=str, help='Choose[ce,gce,sce,boot,focal]')

    # 导入Loss相关的hyper parameters
    loss_name = parser.parse_known_args()[0].loss
    loss_hp_parser = LossHP[loss_name]
    parser = loss_hp_parser.append(parser)

    ##注意一下的argparse和外面main处理bool的方法不一样，感觉下面的方案更简洁一些
    parser.add_argument('--clear_model', action='store_true', default=False, help='Whether to clear existing model')

    parser.add_argument('--use_gpu', action='store_true', default=False, help='Whether to enable gpu')
    parser.add_argument('--device', type=str, default='0', help='which gpu to use')

    parser.add_argument('--do_train', action='store_true', default=False, help='run model training')
    parser.add_argument('--do_export', action='store_true', default=False, help='export model')
    parser.add_argument('--do_eval', action='store_true', default=False, help='run evaluation')
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable debugging logging
    main(args)

