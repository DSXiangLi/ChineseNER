import argparse
import importlib
import pickle

import tensorflow as tf

from config import RUN_CONFIG
from dataset import NerDataset, MultiDataset
from tools.train_utils import build_model_fn,build_mtl_model_fn
from tools.utils import clear_model, build_estimator


def singletask_train(args):
    model_name = args.rename if args.rename else args.model_name
    model_dir = './checkpoint/ner_{}_{}'.format(args.data, model_name)
    data_dir = './data/{}'.format(args.data)
    if args.clear_model:
        clear_model(model_dir)

    # Init dataset and pass parameter to train_params
    TRAIN_PARAMS = getattr(importlib.import_module('model.{}'.format(args.model_name)),'TRAIN_PARAMS')
    input_pipe = NerDataset(data_dir, TRAIN_PARAMS['batch_size'], TRAIN_PARAMS['epoch_size'])
    TRAIN_PARAMS.update(input_pipe.params) # add label_size, max_seq_len, num_train_steps into train_params
    print('='*10+'TRAIN PARAMS'+'='*10)
    print(TRAIN_PARAMS)
    print('='*10+'RUN PARAMS'+'='*10)
    print(RUN_CONFIG)

    # Init estimator'
    model_fn = build_model_fn(args.model_name)
    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn, args.gpu, RUN_CONFIG)

    # Run Train & Evaluate
    early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator, metric_name='loss',
        max_steps_without_decrease=int(TRAIN_PARAMS['step_per_epoch'] * TRAIN_PARAMS['early_stop_ratio'])
    )
    train_spec = tf.estimator.TrainSpec(input_pipe.build_input_fn('train'), hooks=[early_stopping_hook])
    eval_spec = tf.estimator.EvalSpec(input_pipe.build_input_fn('valid', is_predict=True), throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Do prediction when train finished
    prediction = estimator.predict(input_fn=input_pipe.build_input_fn('predict', is_predict=True))
    prediction = [i for i in prediction]
    with open('./data/{}/{}_predict.pkl'.format(args.data, model_name), 'wb') as f:
        pickle.dump(prediction, f)


def multitask_train(args):
    """
    Train Multitask or adversarial task. Must provide more than 1 dataset, and corresponding mtl/adv model
    Only used for train and evaluate, for prediction use above prediciont
    """
    model_name = args.rename if args.rename else args.model_name
    model_dir = './checkpoint/ner_{}_{}'.format('_'.join(args.data.split(',')), model_name)

    if args.clear_model:
        clear_model(model_dir)
    data_dir = './data'
    data_list = args.data.split(',')

    # Init dataset and pass parameter to train_params
    TRAIN_PARAMS = getattr(importlib.import_module('model.{}'.format(args.model_name)),'TRAIN_PARAMS')
    input_pipe = MultiDataset(data_dir, data_list, TRAIN_PARAMS['batch_size'], TRAIN_PARAMS['epoch_size'])
    TRAIN_PARAMS.update(input_pipe.params) # add label_size, max_seq_len, num_train_steps into train_params
    print('='*10+'TRAIN PARAMS'+'='*10)
    print(TRAIN_PARAMS)
    print('='*10+'RUN PARAMS'+'='*10)
    print(RUN_CONFIG)

    # Init estimator
    model_fn = build_mtl_model_fn(args.model_name)
    estimator = build_estimator(TRAIN_PARAMS, model_dir, model_fn, args.gpu, RUN_CONFIG)

    #Run Train & Evaluate
    early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator, metric_name='loss',
        max_steps_without_decrease=int(TRAIN_PARAMS['step_per_epoch'] * TRAIN_PARAMS['early_stop_ratio'])
    )
    train_spec = tf.estimator.TrainSpec(input_pipe.build_input_fn('train'), hooks=[early_stopping_hook])
    eval_spec = tf.estimator.EvalSpec(input_pipe.build_input_fn('valid'), throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #Do prediction when train finished
    for data in data_list:
        print('Prediction for {}'.format(data))
        prediction = estimator.predict(input_pipe.build_predict_fn(data))
        prediction = [pred for pred in prediction]
        # for mutli-task, prediction file name is {model_name}_{task_list}_predict
        with open('./data/{}/{}_{}_predict.pkl'.format(data, model_name,
                                                       '_'.join(args.data.split(','))), 'wb') as f:
            pickle.dump(prediction, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model_name[bert_bilstm_crf, bert_crf, bert_ce]',
                        required=True)
    parser.add_argument('--clear_model', type=int, help='Whether to clear existing model',
                        required=False, default=0)
    parser.add_argument('--data', type=str, help='which data to use[msra, cluener, people_daily]',
                        required=False, default='msra')
    parser.add_argument('--gpu', type=int, help='Whether to enable gpu',
                        required=False, default=0)
    parser.add_argument('--device', type=int, help='which gpu to use',
                        required=False, default=-1)
    parser.add_argument('--rename', type=str, help='Allow rename model with special parameter',
                        required=False, default='')
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)

    if len(args.data.split(','))>1:
        multitask_train(args)
    else:
        singletask_train(args)
