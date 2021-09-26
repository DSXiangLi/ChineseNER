# -*-coding:utf-8 -*-
"""
Lattice is the mix of giga char embedding & ctb50 word embedding, this will be used in the FLAT Lattice Transformer
"""


def combine_w2v(word_path, char_path, output_path):
    # only keep the word in word_model, remove all single character
    # concat with the char_model

    with open(output_path, 'w') as fw:
        counter = 0
        with open(word_path, 'r') as fr:
            for l in fr.readlines():
                word = l.strip().split()[0]
                if len(word)>1:
                    fw.write(l.strip())
                    fw.write('\n')
                    counter+=1
        print('Write {} word embedding to Mix'.format(counter))

        counter = 0
        with open(char_path, 'r') as fr:
            for l in fr.readlines():
                fw.write(l.strip())
                fw.write('\n')
                counter +=1
        print('Write {} char embedding to Mix'.format(counter))
    return

if __name__ =='__main__':
    char_path = './pretrain_model/giga/gigaword_chn.all.a2b.uni.ite50.vec'
    word_path = './pretrain_model/ctb50/ctb.50d.vec'
    output_path = './pretrain_model/lattice/word_char_mix_50d.vec'

    combine_w2v(word_path, char_path, output_path)