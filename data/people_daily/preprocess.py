# -*-coding:utf-8 -*-
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


class PDTFRecord(DumpTFRecord):
    def __init__(self, data_dir, file_name, bert_model_dir, max_seq_len=MAX_SEQ_LEN, tag2idx=TAG2IDX):
        super(PDTFRecord, self).__init__(data_dir, file_name, bert_model_dir, max_seq_len, tag2idx)
        self.mapping = {
            'train': 'train',
            'dev': 'valid',
            'test': 'predict'
        }

    def load_data(self):
        """
        Load line and generate sentence and tag list with char split by ' '
        sentences: ['今 天 天 气 好 ', ]
        tags: ['O O O O O ', ]
        """
        tokens = self.read_text('example.{}'.format(self.file_name))
        sentences = []
        tags = []
        tag = []
        sentence = []
        for i in tokens:
            if i =='':
                sentences.append(' '.join(sentence))
                tags.append(' '.join(tag))
                tag = []
                sentence = []
            else:
                s, t = i.split(' ')
                tag.append(t)
                sentence.append(s)
        return sentences, tags


if __name__ == '__main__':
    data_dir = './data/people_daily'
    bert_model = './pretrain_model/ch_wwm_ext'
    for file in ['train', 'dev', 'test']:
        prep = PDTFRecord(data_dir, file, bert_model)
        prep.dump_tfrecord()
