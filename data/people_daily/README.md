## Chinese NER - People's Daily dataset

人民日报数据集，标注形式为BIO，train20865，dev2318， test4636条语料
包括 LOC,ORG,PER三类实体

1. 原始数据包括exmaple.dev, example.train, example.test三个file
2. 运行python ./data/people_daily/preprocess.py 生成原始tfrecord和data_params数据，以及使用3种word_enhance的tfrecord

**Reference**:   
<https://github.com/zjy-ucas/ChineseNER>