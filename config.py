# -*-coding:utf-8 -*-
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

TRAIN_PARAMS = {
    'dtype': tf.float32,
    'lr': 5e-6,
    'log_steps': 100,
    'pretrain_dir': './pretrain_model/ch_google', # pretrain Bert-Model
    'batch_size': 32,
    'epoch_size': 20,
    'embedding_dropout': 0.1,
    'warmup_ratio': 0.1,
    'early_stop_ratio': 1 # stop after ratio * steps_per_epoch
}


RUN_CONFIG = {
    'summary_steps': 10,
    'log_steps': 100,
    'save_steps': 500,
    'keep_checkpoint_max': 3,
    'allow_growth': True,
    'pre_process_gpu_fraction': 0.8,
    'log_device_placement': True,
    'allow_soft_placement': True,
    'inter_op_parallel': 2,
    'intra_op_parallel': 2
}

