## MRC框架

因为和传统的序列标注任务输入不同索性单独写了一个模块
1. 数据样本准备:把MSRA，people_daily样本转换成MRC样本
```bash 
python mrc/convert2mrc.py
```
2. 训练：
   - 训练完会直接Eval，并把Evaluation写进对应checkpoint里面的train.log
   - 预测结果和抽取的实体会分别写两个文件到对应data_dir
```bash 
python mrc/main.py --data msra --clear_model --do_train --do_eval --do_export
```
