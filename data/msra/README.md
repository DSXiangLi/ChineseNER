## Chinese NER Dataset -MSRA

微软亚研院MSRA命名实体识别识别数据集

标注形式为BIO，共有46365条语料。包括ORG，LOC, PER三类实体

1. 原始数据包括train,eval,test三个folder. Train 42000条，Validate 3000条，Test 3442条。
2. 运行python ./data/msra/preprocess.py 生成原始tfrecord和data_params数据，以及使用3种word_enhance的tfrecord

**Reference**:   
<https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra>