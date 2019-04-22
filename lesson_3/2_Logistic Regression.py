#!/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import random

class Logistic_Regression():
    def __init__(self):
        pass

    #推理
    def inference(self, w, b, x):
        h = w * x + b
        pred_y = 1 / (1 + np.exp(-h))
        return  pred_y

    #损失
    def eval_loss(self, x_list, gt_y_list, w, b):
        eval_loss = 0.0

        for i in range(len(x_list)):
            h = w * x_list[i] + b
            pred_y = 1 / (1 + np.exp(-h))
            eval_loss += -1 * gt_y_list[i] * np.log(pred_y) - (1 - gt_y_list[i]) * np.log(1 - pred_y)

        eval_loss /= len(gt_y_list)
        return eval_loss

    #参数更新量(每个样本)
    def gradient(self, pred_y, gt_y, x):
        diff = pred_y - gt_y
        dw = x * diff
        db = diff
        return dw, db

    #参数更新
    def cal_step_gradient(self, x_batch, gt_y_batch, w, b, lr):
        avg_dw, avg_db = 0, 0
        batch_size = len(gt_y_batch)

        for i in range(batch_size):
            pred_y = self.inference(w, b, x_batch[i])
            dw, db = self.gradient(pred_y, gt_y_batch[i], x_batch[i])
            avg_dw += dw
            avg_db += db

        avg_dw /= batch_size
        avg_db /= batch_size

        w -= lr * avg_dw
        b -= lr * avg_db

        return w, b

    #训练
    def train(self, x_list, gt_y_list, batch_size, lr, max_iter):
        w, b = 0, 0
        num_samples = len(gt_y_list)

        for i in range(max_iter):
            batch_idxs = np.random.choice(len(x_list), batch_size)
            x_batch = [x_list[i] for i in batch_idxs]
            y_batch = [gt_y_list[i] for i in batch_idxs]
            w, b = self.cal_step_gradient(x_batch, y_batch, w, b, lr)
            print('w:{0}, b:{1}'.format(w, b))
            print('loss is {0}'.format(self.eval_loss(x_list, gt_y_list, w, b)))

    #数据生成
    def gen_data(self):
        w = random.randint(0, 10) + random.random()
        b = random.randint(0, 5) + random.random()

        num_samples = 100
        x_list = []
        gt_y_list = []

        for i in range(num_samples):
            x = random.randint(0, 100) * random.random()
            y = 1 if x > 50 else 0
            x_list.append(x)
            gt_y_list.append(y)
        return x_list, gt_y_list, w, b

    #运行
    def run(self):
        x_list, gt_y_list, w, b = self.gen_data()
        batch_size = 50
        lr = 0.01
        max_iter = 10000
        self.train(x_list, gt_y_list, batch_size, lr, max_iter)

if __name__ == '__main__':
    LR = Logistic_Regression()
    LR.run()