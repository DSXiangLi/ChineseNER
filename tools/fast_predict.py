# -*-coding:utf-8 -*-
"""
    刚发现的宝藏方案。Estiamtor.predict每次调用都会重新加载计算图，如果在需要多次调用的场景（数据交互/serving）耗时非常高，
    以下通过generator的暂停效果，hold住predict不退出，predict速度原地起飞

    Credit to: https://github.com/marcsto/rl/blob/master/src/fast_predict.py
"""


class FastPredict:
    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def stream_predict(self, feature):
        """
        这了改成了单条样本的预测，原版是batch，但是都交互了哈哈其实是很少用batch的
        """
        self.next_features = feature
        if self.first_run:
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator))
            self.first_run = False

        return next(self.predictions)

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")
