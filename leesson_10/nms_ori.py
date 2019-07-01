#!/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
dets = np.array([ [204, 102, 358, 250, 0.5],
                [257, 118, 380, 250, 0.7],
                [280, 135, 400, 250, 0.6],
                [255, 118, 360, 235, 0.7]])
thresh = 0.55

x1  = dets[:, 0]
y1 = dets[:, 1]
x2 = dets[:, 2]
y2 = dets[:, 3]
scores = dets[:, 4]

areas = (x2 - x1 + 1) * (y2 - y1 + 1)   #四个box的面积
order = scores.argsort()[::-1]      #[3, 1, 2, 0]按照score从大到小返回索引

keep = []       #存放保留下来的框的索引
while order.size > 0:
    i = order[0]    #score最大的索引
    keep.append(i)

    #获取当前框与所有框的重叠部分
    xx1 = np.maximum(x1[i], x1[order[1:]])      #返回x1[i]与x1[order[1:]]中较大的值
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])      #返回x2[i]与x2[order[1:]]中较小的值
    yy2 = np.minimum(y2[i], y2[order[1:]])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h       #重叠部分面积

    ovr = inter / (areas[i] + areas[order[1:]] - inter) #计算重叠度
    inds = np.where(ovr <= thresh)[0]       #np.where(condition),返回满足条件的索引.  这里返回重叠度小于阈值的索引,即要保留的框的索引。
    order = order[inds + 1]     #原order中的首位置所代表的框已经不要了，剩余索引往后推一位。
print(keep)