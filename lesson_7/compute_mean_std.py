#!/user/bin/env python3
# -*- coding:utf-8 -*-

'''
计算3个通道均值方差
normMeans = [0.437, 0.435, 0.434]
normStd = [0.303, 0.300, 0.301]
'''
import numpy as np
import random
import cv2 as cv
train_txt_path = 'data/train.txt'
img_h, img_w = 640, 640
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)

    for i in range(len(lines)):
        img_path = lines[i].rstrip().split()[0] + ' ' + lines[i].rstrip().split()[1]
        # print(img_path)
        img = cv.imread(img_path)
        img = cv.resize(img, (img_h, img_w))

        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis = 3)

imgs = imgs.astype(np.float32) / 255


for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append((np.std(pixels)))

means.reverse()
stdevs.reverse()

print('normMeans = {}'.format(means))
print(('normStd = {}'.format(stdevs)))