# -*-coding:utf-8 -*-
import grpc
import tensorflow as tf
import numpy as np
import re
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

from data.base_preprocess import get_instance, extract_prefix_surfix
from data.tokenizer import TokenizerBert
from tools.infer_utils import extract_entity, fix_tokens, timer, grpc_retry

MODEL = 'bert_bilstm_crf_mtl'
SERVER = 'localhost:8500'
VERSION = 1
MAX_SEQ_LEN = 150
TIMEOUT = 10

TAG2IDX = {
    '[PAD]': 0,
    'O': 1,
    'B-ORG': 2,
    'I-ORG': 3,
    'B-PER': 4,
    'I-PER': 5,
    'B-LOC': 6,
    'I-LOC': 7,
    '[CLS]': 8,
    '[SEP]': 9
}


class InferHelper(object):
    def __init__(self, max_seq_len, tag2idx, model_name, version, server, timeout):
        self.model_name = model_name
        self.word_enhance, self.tokenizer_type = extract_prefix_surfix(model_name)
        self.mtl = 1 if re.search('(mtl)|(adv)', model_name) else 0  # whether is multitask
        self.proc = get_instance(self.tokenizer_type, max_seq_len, tag2idx,
                                 word_enhance=self.word_enhance, mapping=None)
        self.max_seq_len = max_seq_len
        self.tag2idx = tag2idx
        self.idx2tag = dict([(val, key) for key, val in tag2idx.items()])
        self.server = server
        self.version = version
        self.timeout = timeout

    def get_stub(self):
        channel = grpc.insecure_channel(self.server)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        return stub

    def make_request(self, feature):
        request = predict_pb2.PredictRequest()
        request.model_spec.signature_name = 'serving_default' # set in estimator output
        request.model_spec.name = self.model_name
        request.model_spec.version.value = self.version
        tensor_proto = tensor_util.make_tensor_proto(feature, dtype=tf.string)
        request.inputs['example'].CopyFrom(tensor_proto)
        return request

    def make_feature(self, sentence):
        self.feature = self.proc.build_seq_feature(sentence)
        # fake labels and label_ids, if you want to skip this you need to modify model_fn
        self.feature['labels'] = np.zeros(shape=(self.max_seq_len,)).astype(str).tolist()
        self.feature['label_ids'] = np.zeros(shape=(self.max_seq_len,)).astype(int).tolist()

        if self.mtl:
            self.feature['task_ids'] = 1

        if self.tokenizer_type == TokenizerBert:
            # fix word piece tokenizer UNK and ##
            self.feature['tokens'] = fix_tokens(sentence, self.feature['tokens'])

        tf_example = tf.train.Example(
                        features=tf.train.Features(feature=self.proc.build_tf_feature(self.feature))
                    )
        return [tf_example.SerializeToString()]

    @grpc_retry()
    def decode_prediction(self, resp):
        res = resp.result().outputs
        pred_ids = np.squeeze(tf.make_ndarray(res['pred_ids'])) # seq label ids
        entity = extract_entity(self.feature['tokens'], pred_ids, self.idx2tag)
        return entity

    @timer
    def infer(self, text):
        stub = self.get_stub()
        feature = self.make_feature(text)
        req = self.make_request(feature)
        resp = stub.Predict.future(req, self.timeout)
        output = self.decode_prediction(resp)
        return output


# create singleton for inference, trigger all lazy eval before online inference
infer_handle = InferHelper(MAX_SEQ_LEN, TAG2IDX, MODEL, VERSION, SERVER, timeout=TIMEOUT)


if __name__ == '__main__':
    print('\n Input text for inference, press Enter when finished \n')
    while True:
        text = input('Input Text: ')
        print(infer_handle.infer(text))