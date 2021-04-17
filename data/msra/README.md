## Chinese NER Dataset -MSRA

微软亚研院MSRA命名实体识别识别数据集

标注形式为BIO，共有46365条语料。包括ORG，LOC, PER三类实体

1. 下载数据集到./data/msra, 包括train,eval,test三个folder
2. python ./data/msra/preprocess.py 生成tfrecord和data_params数据

**Reference**:   
<https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra>