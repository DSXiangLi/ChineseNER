# -*-coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow_serving.apis import prediction_log_pb2

from inference import InferHelper, MAX_SEQ_LEN, TAG2IDX, MODEL, VERSION, SERVER, TIMEOUT, MTL

NUM_RECORDS=5


def main():

    """Generate TFRecords for warming up."""
    infer_handle = InferHelper(MAX_SEQ_LEN, TAG2IDX, MODEL, VERSION, SERVER, timeout=TIMEOUT, mtl=MTL)
    warmup_text =  '给中央军委委员、总参谋长傅全有上将致唁函的有:美国太平洋总部司令布鲁赫海军上将。'

    os.mkdir('./serving_model/{}/{}/assets.extra'.format(MODEL, VERSION))
    with tf.io.TFRecordWriter("./serving_model/{}/{}/assets.extra/tf_serving_warmup_requests".format(MODEL, VERSION)) as writer:
        feature = infer_handle.make_feature(warmup_text)
        req = infer_handle.make_request(feature)
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=req))
        for r in range(NUM_RECORDS):
            writer.write(log.SerializeToString())


if __name__ == '__main__':
    main()