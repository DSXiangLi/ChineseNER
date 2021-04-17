# -*-coding:utf-8 -*-

import os
from data.base_preprocess import DumpTFRecord

TAG2IDX = {
    '[PAD]': 0,
    'B': 1,
    'I': 2,
    'E': 3,
    'S': 4,
    '[CLS]': 5,
    '[SEP]': 6
}

MAX_SEQ_LEN = 150


class MsrTFRecord(DumpTFRecord):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len=MAX_SEQ_LEN, tag2idx=TAG2IDX):
        super(MsrTFRecord, self).__init__(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx)
        self.mapping = {
            'training': 'train',
            'test_gold': 'valid',
            'test': 'predict'
        }

    def load_data(self):
        """
        Load line, word are separated by ' ', use len of tokenizer to avoid mismatch with bert tokenizer
        """
        lines = self.read_text('msr_{}.utf8'.format(self.file_name))
        sentences = []
        tags = []
        for line in lines:
            if line == '':
                continue
            else:
                tag = [self.gen_tag(len(self.tokenizer.tokenize(token))) for token in line.split(' ') if token not in ['', '"']]
                sentence = [token for token in line.split(' ') if token not in ['', '"']]
            tags.append(' '.join(tag))
            sentences.append(' '.join(sentence))
        return sentences, tags

    @staticmethod
    def gen_tag(length):
        if length == 1:
            return 'S'
        elif length ==2:
            return 'B E'
        else:
            return ' '.join(['B'] + ['I'] * (length-2) + ['E'] )


if __name__ == '__main__':
    data_dir = './data/msr'
    bert_model = './pretrain_model/ch_wwm_ext'
    for file in ['training', 'test_gold', 'test']:
        prep = MsrTFRecord(data_dir, file, bert_model)
        prep.dump_tfrecord()
