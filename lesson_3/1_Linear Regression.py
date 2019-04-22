#!/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import random

class Linear_Regression():
     def __init__(self):
         pass

    #推理函数
     def inference(self, w, b, x):
         pred_y = w * x + b
         return pred_y

     #损失
     def eval_loss(self, w, b, x_list, gt_y_list):
         eval_loss = 0.0

         #loop
         # for i in range(len(x_list)):
         #     eval_loss += 0.5 * (gt_y_list[i] - (w * x_list[i] + b)) ** 2

         #python way
         eval_loss = np.sum(0.5 * (gt_y_list - (w * x_list + b)) ** 2)
         eval_loss /= len(x_list)
         return eval_loss

     #梯度
     def gradient(self, pred_y, gt_y_list, x):
         diff = pred_y - gt_y_list
         dw = diff * x
         db = diff
         return dw, db

     #参数更新（1个batch更新1次）
     def cal_step_gradient(self, batch_x_list, gt_batch_y_list, w, b, lr):
         avg_w, avg_b = 0, 0
         batch_size = len(gt_batch_y_list)
         for i in range(len(gt_batch_y_list)):
             pred_y = self.inference(w, b, batch_x_list[i])
             dw, db = self.gradient(pred_y, gt_batch_y_list[i], batch_x_list[i])
             avg_w += dw   #每个样本更新的参数改变量加起来
             avg_b += db
         avg_w /= batch_size    #除以batch得到平均参数改变量
         avg_b /= batch_size

         w -= lr * avg_w   #更新参数
         b -= lr * avg_b
         return w, b

     #训练
     def train(self, x_list, gt_y_list, batch_size, lr, max_iter):
         w, b = 0, 0
         for _ in range(max_iter):
             batch_index = np.random.choice(len(gt_y_list), batch_size)
             batch_x = [x_list[i] for i in batch_index]
             batch_y = [gt_y_list[i] for i in batch_index]
             w, b = self.cal_step_gradient(batch_x, batch_y, w, b, lr)
             loss = self.eval_loss(w, b, x_list, gt_y_list)
             print('w:{0}, b:{1}'.format(w, b))
             print('loss is {0}'.format(loss))

     #产生数据
     def gen_data(self):
         w = random.randint(0, 10) + random.random()
         b = random.randint(0, 5) + random.random()
         num_samples = 100

         #loop
         # x_list = []
         # gt_y_list = []
         # for i in range(num_samples):
         #     x = random.randint(0, 100) * random.random()
         #     y = w * x + b + random.random() * random.randint(-1, 1)
         #     x_list.append(x)
         #     gt_y_list.append(y)

         #python way
         x_list = np.random.randint(0, 100, num_samples) * random.random()
         gt_y_list = w * x_list + b + random.random() * random.randint(-1, 1)

         return x_list, gt_y_list, w, b

     #运行
     def run(self):
         x_list, gt_y_list, w, b = self.gen_data()
         max_iter = 10000
         lr = 0.001
         batch_size = 50
         self.train(x_list, gt_y_list, batch_size, lr, max_iter)

if __name__ == '__main__':
    L_R = Linear_Regression()
    L_R.run()

