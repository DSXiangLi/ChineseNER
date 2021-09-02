## 预训练模型

下载预训练模型到当前Folder，受文件大小限制这里移除了checkpoint文件，请自行去以下链接下载完整模型
哈工大wwm bert的vocab文件和google bert一致，所以tokenizer加载哪个模型结果都是一样的。

1. ch_google: Google bert base chinese，L-12_H-768
- 项目链接：https://github.com/google-research/bert
- 模型链接：https://github.com/qiufengyuyi/sequence_tagging/tree/master/bert

2. ch_wwm_ext: 哈工大BERT-wwm-ext, Chinese，ext, L-12_H-768
- 项目链接： https://github.com/ymcui/Chinese-BERT-wwm
- 模型链接：https://drive.google.com/file/d/1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi/view

3. sgns: People's Daily News 人民日报 300d word+Ngram 预训练词向量
- 项目链接：https://github.com/Embedding/Chinese-Word-Vectors
- 模型链接：https://pan.baidu.com/s/1upPkA8KJnxTZBfjuNDtaeQ

4. Giga: Character embeddings/bichar embedding (gigaword_chn.all.a2b.uni[bi].ite50.vec) 
是Glove word2vec格式想用gensim word2vec加载需要运行glova_2_wv中的covert
- 模型链接：https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing
- 模型链接2： https://pan.baidu.com/s/1pLO6T9D#list/path=%2F
- 项目链接：https://github.com/jiesutd/LatticeLSTM 复用LatticeLSTM

5. ctb50: Word(Lattice) embeddings (ctb.50d.vec)
- 模型链接：https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view
- 项目链接：https://github.com/jiesutd/LatticeLSTM 复用LatticeLSTM
