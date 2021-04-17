# -*-coding:utf-8 -*-
import os
import importlib
import tensorflow as tf
from tensorflow.python.framework import ops
from bert_base.bert import modeling
from bert_base.bert.optimization import AdamWeightDecayOptimizer
from itertools import chain
from tools.utils import get_log_hook


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        self.num_calls += 1

        return y

def token2sequence(tokens):
    """
    convert list ot token_id to string, for inference and tf.summary
    """
    token2str = lambda x: ' '.join([i.decode('utf-8') for i in x])
    return tf.py_func(token2str, [tokens], tf.string)


def id2sequence(mapping):
    def helper(pred_ids):
        """
        convert list ot token_id to string, for inference and tf.summary
        """
        idx2str = lambda x: ' '.join([mapping[i] for i in x])
        return tf.py_func(idx2str, [pred_ids], tf.string)
    return helper


def load_bert_checkpoint(pretrain_dir):
    """
    Load pretrain bert checkpoint, and init train_vars with checkpoint
    """
    ckpt_path = os.path.join(pretrain_dir, 'bert_model.ckpt')

    tvars = tf.trainable_variables()
    assignment_map, init_var = modeling.get_assignment_map_from_checkpoint(
        tvars, ckpt_path)

    tf.train.init_from_checkpoint(ckpt_path, assignment_map)
    return


def get_eval_metrics(label_ids, pred_ids, idx2tag, task_name=''):
    """
    Overall accuracy, and accuracy per tag
    """
    real_length = tf.reduce_sum(tf.sign(label_ids), axis=1) - 2
    max_length = label_ids.shape[-1].value
    mask = tf.sequence_mask(real_length, maxlen=max_length)
    pred_ids = tf.cast(pred_ids, tf.int32)
    if task_name:
        metric_op = {
            'metric_{}/overall_accuracy'.format(task_name): tf.metrics.accuracy(labels=label_ids, predictions=pred_ids, weights=mask)
        }
    else:
        metric_op = {
            'metric/overall_accuracy': tf.metrics.accuracy(labels=label_ids, predictions=pred_ids, weights=mask)
        }
    # add accuracy metric per NER tag
    for id, tag in idx2tag.items():
        id = tf.cast(id, tf.int32)
        metric_op.update(calc_metrics(tf.equal(label_ids, id), tf.equal(pred_ids, id), mask, tag, task_name))

    return metric_op


def calc_metrics(label_ids, pred_ids, weight, prefix, task_name):
    precision, precision_op = tf.metrics.precision(
            labels=label_ids, predictions=pred_ids, weights=weight)
    recall, recall_op = tf.metrics.recall(
            labels=label_ids, predictions=pred_ids, weights=weight)
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics = {
        'metric_{}/{}_accuracy'.format(task_name, prefix): tf.metrics.accuracy(
            labels=label_ids, predictions=pred_ids, weights=weight),
        'metric_{}/{}_precision'.format(task_name, prefix): (precision, precision_op),
        'metric_{}/{}_recall'.format(task_name, prefix): (recall, recall_op),
        'metric_{}/{}_f1'.format(task_name, prefix): (f1, tf.identity(f1))
    }
    return metrics


def build_model_fn(model_name):
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # get model graph given above model_name
        build_graph = getattr(importlib.import_module('model.{}'.format(model_name)), 'build_graph')
        loss, pred_ids, task_ids = build_graph(features=features, labels=labels, params=params, is_training=is_training)

        if is_training:
            tf.summary.text('tokens', token2sequence(features['tokens'][0, :]))
            tf.summary.text('labels', token2sequence(features['labels'][0, :]))

            train_op = create_train_op(loss, params['lr'], params['num_train_steps'],
                                       params['warmup_ratio'], params['diff_lr_times'])
            spec = tf.estimator.EstimatorSpec(mode, loss=loss,
                                              train_op=train_op,
                                              training_hooks=[get_log_hook(loss, params['log_steps'])])
        elif mode == tf.estimator.ModeKeys.EVAL:
            metric = get_eval_metrics(features['label_ids'], pred_ids, params['idx2tag'])
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric)

        else:
            #For serving, only keep pred_ids in inference
            spec = tf.estimator.EstimatorSpec(mode, {'pred_ids': pred_ids,
                                                     'label_ids': features['label_ids'],
                                                     'tokens': features['tokens']
                                                     })
        return spec
    return model_fn


def build_mtl_model_fn(model_name):
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # get model graph given above model_name
        build_graph = getattr(importlib.import_module('model.{}'.format(model_name)), 'build_graph')
        loss, pred_ids, task_ids = build_graph(features=features, labels=labels, params=params, is_training=is_training)
        mask1 = tf.equal(task_ids, 0) # get sample for task1
        mask2 = tf.equal(task_ids, 1) # get sample for task2
        if is_training:
            tokens = tf.boolean_mask(features['tokens'], mask1, axis=0)[0,:]
            labels = tf.boolean_mask(features['labels'], mask1, axis=0)[0,:]
            tf.summary.text('tokens_{}'.format(params['task_list'][0]), token2sequence(tokens))
            tf.summary.text('labels_{}'.format(params['task_list'][0]), token2sequence(labels))

            tokens = tf.boolean_mask(features['tokens'], mask2, axis=0)[0,:]
            labels = tf.boolean_mask(features['labels'], mask2, axis=0)[0,:]
            tf.summary.text('tokens_{}'.format(params['task_list'][1]), token2sequence(tokens))
            tf.summary.text('labels_{}'.format(params['task_list'][1]), token2sequence(labels))

            train_op = create_train_op(loss, params['lr'], params['num_train_steps'],
                                       params['warmup_ratio'], params['diff_lr_times'], True)
            spec = tf.estimator.EstimatorSpec(mode, loss=loss,
                                              train_op=train_op,
                                              training_hooks=[get_log_hook(loss, params['log_steps'])])
        elif mode == tf.estimator.ModeKeys.EVAL:
            metric_op = get_eval_metrics(tf.boolean_mask(features['label_ids'], mask1),
                                       tf.boolean_mask(pred_ids, mask1),
                                       params[params['task_list'][0]]['idx2tag'], task_name=params['task_list'][0])
            metric_op.update(get_eval_metrics(tf.boolean_mask(features['label_ids'], mask2),
                                       tf.boolean_mask(pred_ids, mask2),
                                       params[params['task_list'][1]]['idx2tag'], task_name=params['task_list'][1]))
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_op)
        else:
            #For serving, only keep pred_ids in inference
            spec = tf.estimator.EstimatorSpec(mode, {'pred_ids': pred_ids,
                                                     'label_ids': features['label_ids'],
                                                     'tokens': features['tokens']
                                                     })
        return spec
    return model_fn


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, global_step):
    """
    Basic optimizer from bert_base, including linear warm up and exponential decay
    """
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    return optimizer, learning_rate


def create_train_op(loss, init_lr, num_train_steps, warmup_ratio, diff_lr_times, verbose=False):
    """
    Allow using different learning rate schema for different layer. If not this is the same as bert pretrain opt/lr
    """
    global_step = tf.train.get_or_create_global_step()
    tvars = tf.trainable_variables()

    num_warmup_steps = int(num_train_steps * warmup_ratio)

    if diff_lr_times:
        opt_list = []
        var_list = []
        # lr/opt for other layers
        for name, times in diff_lr_times.items():
            opt, lr = create_optimizer(init_lr * times, num_train_steps, num_warmup_steps, global_step)
            opt_list.append(opt)
            var_list.append([i for i in tvars if name in i.name])
            tf.summary.scalar('lr/{}'.format(name), lr)
        # basic lr/opt for bert mainly
        opt, lr = create_optimizer(init_lr, num_train_steps, num_warmup_steps,
                                   global_step)
        vars = [i for i in tvars if i not in list(chain(*var_list))]
        opt_list.append(opt)
        var_list.append(vars)
        tf.summary.scalar('lr/org', lr)

        # calculate gradient for all vars and clip gradient
        all_grads = tf.gradients(loss, list(chain(*var_list)))
        (all_grads, _) = tf.clip_by_global_norm(all_grads, clip_norm=1.0)
        if verbose:
            for var, grad in zip(list(chain(*var_list)), all_grads):
                if (grad is not None) and 'share_max_pool' in var.name:
                    tf.summary.histogram('grad/{}'.format(var.name), grad)

        # back propagate given different learning rate
        train_op_list = []
        for vars, opt in zip(var_list, opt_list):
            num_vars = len(vars)
            grads = all_grads[:num_vars]
            all_grads = all_grads[num_vars:]
            train_op = opt.apply_gradients(zip(grads, vars), global_step=global_step)
            train_op_list.append(train_op)
        train_op = tf.group(train_op_list, [global_step.assign(global_step + 1)])
    else:
        opt, lr = create_optimizer(init_lr, num_train_steps, num_warmup_steps, global_step)
        tf.summary.scalar('lr', lr)
        grads = tf.gradients(loss, tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        train_op = tf.group(train_op, [global_step.assign(global_step + 1)])
    return train_op