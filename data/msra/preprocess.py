# -*-coding:utf-8 -*-
import os
from data.base_preprocess import DumpTFRecord

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

MAX_SEQ_LEN = 150


class MsraTFRecord(DumpTFRecord):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len=MAX_SEQ_LEN, tag2idx=TAG2IDX):
        super(MsraTFRecord, self).__init__(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx)
        self.mapping = {
            'train': 'train',
            'val': 'valid',
            'test': 'predict'
        }

    def load_data(self):
        """
        Load sentence and tags
        """
        sentences = self.read_text(os.path.join(self.file_name,'sentences.txt'))
        tags = self.read_text(os.path.join(self.file_name, 'tags.txt'))
        assert len(sentences) == len(tags)
        return sentences, tags


if __name__ == '__main__':
    data_dir = './data/msra'
    bert_model = './pretrain_model/ch_wwm_ext'
    for file in ['train', 'val', 'test']:
        prep = MsraTFRecord(data_dir, file, bert_model)
        prep.dump_tfrecord()
