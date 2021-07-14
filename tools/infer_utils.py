# -*-coding:utf-8 -*-
import time
import tensorflow as tf
from functools import wraps
from collections import defaultdict
from grpc import StatusCode, RpcError

from data.base_preprocess import get_feature_poroto

EXPORT_DIR = './serving_model/{}'

MsTime = lambda: int(round(time.time() * 1000))

RETRY_TIEMS = {
    StatusCode.INTERNAL: 1,
    StatusCode.ABORTED: 3,
    StatusCode.UNAVAILABLE: 3,
    StatusCode.DEADLINE_EXCEEDED: 5  # most-likely grpc channel close, need time to reopen
}


def grpc_retry(default_max_retry=3, sleep=0.01):
    def helper(func):
        @wraps(func)
        def handle_args(*args, **kwargs):
            counter = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except RpcError as e:
                    max_retry = RETRY_TIEMS.get(e.code(), default_max_retry)
                    counter += 1
                    if counter >= max_retry:
                        raise e
                    print('Capturing {} , retrying for {} times'.format(e, counter))
                    time.sleep(sleep)  # wait for grpc to reopen channel

        return handle_args

    return helper


def timer(func):
    @wraps(func)
    def helper(*args, **kwargs):
        start_ts = MsTime()
        output = func(*args, **kwargs)
        end_ts = MsTime()
        print('{} latency = {}'.format(func.__name__, end_ts - start_ts))
        return output

    return helper


def get_receiver(max_seq_len, word_enhance, mtl=False):
    # get tf_proto given difference max_seq_len, word_enhance
    def serving_input_receiver_fn():
        tf_proto = get_feature_poroto(max_seq_len, word_enhance)
        if mtl:
            tf_proto.update({
                'task_ids': tf.io.FixedLenFeature([], dtype=tf.int64)
            })
        serialized_tf_example = tf.compat.v1.placeholder(
            dtype=tf.dtypes.string,
            shape=[None],
            name='input_tensor')
        receiver_tensors = {'example': serialized_tf_example}
        features = tf.compat.v1.io.parse_example(serialized_tf_example,
                                                 tf_proto)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    return serving_input_receiver_fn


def extract_entity(tokens, pred_ids, idx2tag):
    """
    Extract different types of entity from sequence label
    """
    assert len(tokens) == len(pred_ids), '{}!={} tokens and pred_ids must have same length'.format(len(tokens),
                                                                                                   len(pred_ids))
    ngram = ''
    entity = defaultdict(set)
    prev_tag = idx2tag[pred_ids[0]]
    for i, (t, id) in enumerate(zip(tokens, pred_ids)):
        if idx2tag[id].split('-')[0] == 'I':
            if prev_tag[0].split('-')[0] in ['B', 'I']:
                ngram += t
        else:
            if ngram != '':
                entity[prev_tag.split('-')[1]].add(ngram)
            if idx2tag[id].split('-')[0] == 'B':
                ngram = t
            else:
                ngram = ''
        prev_tag = idx2tag[id]
    if ngram != '':
        entity[prev_tag.split('-')[1]].add(ngram)
    return entity


def fix_tokens(sentence, tokens):
    """
    Fix ## and UNK in word piece tokenizer, make sure docode has right character
    """
    j = 0
    for i in range(len(tokens)):
        if tokens[i] == '[UNK]':
            tokens[i] = sentence[j]
            j += 1
        elif tokens[i][:2] == '##':
            tokens[i] = tokens[i].replace('##', '')
            j += len(tokens[i])
        elif tokens[i] in ['[PAD]', '[CLS]', '[SEP]']:
            continue
        else:
            j += len(tokens[i])
    return tokens
