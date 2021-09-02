# -*-coding:utf-8 -*-

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path
import os


def convert(model_name):
    abs_path = Path(__file__).absolute().parent

    glove_file = datapath(os.path.join(abs_path, model_name))
    tmp_file = get_tmpfile(os.path.join(abs_path, model_name + 'tmp'))
    if os.path.isfile(os.path.join(abs_path, tmp_file)):
        print('Tmp file already existed, only generate tmp file 1 time')
    else:
        _ = glove2word2vec(glove_file, tmp_file)

    model = KeyedVectors.load_word2vec_format(tmp_file)
    return model