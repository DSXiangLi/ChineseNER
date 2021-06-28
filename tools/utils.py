# -*-coding:utf-8 -*-
import os
import numpy as np
import shutil
import tensorflow as tf


def normalize(vector: np.ndarray):
    norm = np.linalg.norm(vector)
    if norm == 0:
        norm = np.finfo(vector.dtype).eps
    return vector / norm


def clear_model(model_dir):
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))
    else:
        print( '{} model cleaned'.format(model_dir) )


def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag.replace(':', '_')), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag.replace(':', '_')), value)


def build_estimator(params, model_dir, model_fn, gpu_enable, RUN_CONFIG):
    session_config = tf.ConfigProto()

    if gpu_enable:
        # control CPU and Mem usage
        session_config.gpu_options.allow_growth = RUN_CONFIG['allow_growth']
        session_config.gpu_options.per_process_gpu_memory_fraction = RUN_CONFIG['pre_process_gpu_fraction']
        session_config.log_device_placement = RUN_CONFIG['log_device_placement']
        session_config.allow_soft_placement = RUN_CONFIG['allow_soft_placement']
        session_config.inter_op_parallelism_threads = RUN_CONFIG['inter_op_parallel']
        session_config.intra_op_parallelism_threads = RUN_CONFIG['intra_op_parallel']

    run_config = tf.estimator.RunConfig(
        save_summary_steps=RUN_CONFIG['summary_steps'],
        log_step_count_steps=RUN_CONFIG['log_steps'],
        keep_checkpoint_max=RUN_CONFIG['keep_checkpoint_max'],
        save_checkpoints_steps=RUN_CONFIG['save_steps'],
        session_config=session_config, eval_distribute=None
    )
    if os.path.isdir(model_dir):
        warm_start_dir = model_dir
    else:
        warm_start_dir = None

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params,
        model_dir=model_dir,
        warm_start_from=warm_start_dir
    )
    return estimator


def get_log_hook(loss, save_steps):
    hook = {}
    hook['global_step'] = tf.train.get_or_create_global_step()
    hook['loss'] = loss
    log_hook = tf.train.LoggingTensorHook(hook, every_n_iter=save_steps)
    return log_hook


def get_pr_hook(logits, labels, output_dir, save_steps, prefix):
    # add precision-recall curve summary
    pr_summary = tf.summary.pr_curve( name='{}_pr_curve'.format(prefix),
                                      predictions=tf.sigmoid( logits ),
                                      labels=tf.cast( labels, tf.bool ),
                                      num_thresholds= 20 )

    summary_hook = tf.train.SummarySaverHook(
        save_steps= save_steps,
        output_dir= output_dir,
        summary_op=[pr_summary]
    )

    return summary_hook