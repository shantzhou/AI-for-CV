#!/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


class K_means():
    # 1.产生数据
    def gen_data(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        data = X[:, [1, 3]]
        return data

    # 2.初始化中心点
    def random_center(self, data, k):
        n = data.shape[1]  # 中心点维度
        centroids = np.zeros((k, n))
        for i in range(n):
            dmin, dmax = np.min(data[:, i]), np.max(data[:, i])
            centroids[:, i] = dmin + (dmax - dmin) * np.random.random(k)  # 这行代码值得推敲，一行产生两个数，由random得到不同的两个数
        return centroids

    # 3.欧式距离
    def _distance(self, p1, p2):
        dist = np.sum((p1 - p2) ** 2)
        dist = np.sqrt(dist)
        return dist

    # 4.收敛条件(判断centroids是否更新)
    def converged(self, old_centroids, centroids):
        set1 = set([tuple(p) for p in old_centroids])
        set2 = set([tuple(p) for p in centroids])
        return (set1 == set2)

    # 5.k_means
    def k_means(self, data, k):
        #初始化
        n = data.shape[0]  # 样本个数
        centroids = self.random_center(data, k)  # 初始化质心
        label = np.zeros(n, dtype=np.int)  # 初始化样本标签,所以注明int型
        converge = False
        assenment = np.zeros(n)  # 里面保存各样本点到其簇中心的距离，通过距离之和判别聚类的好坏
        #聚类
        while not converge:
            # 计算每个样本点到各个质心的距离
            min_dist, min_idx = np.inf, -1
            for i in range(n):  # 遍历样本
                old_centroids = np.copy(centroids)  # 备份原质心便于比较
                for j in range(k):  # 遍历质心
                    dist = self._distance(data[i], centroids[j])
                    if dist > min_dist:
                        min_dist, min_idx = dist, j
                        label[i] = j
                assenment[i] = self._distance(data[i], centroids[label[i]])  # 推敲，label[i]表示第i个样本所属类别，centroids[label[i]]表示对应的类别簇中心
            # 更新质心
            for i in range(data.shape[1]):
                centroids[:, i] = np.mean(data[label == i], axis=0)  # 对列求均值
            converge = self.converged(old_centroids, centroids)
        return centroids, label, np.sum(assenment)


if __name__ == '__main__':
    k_means_ = K_means()
    data = k_means_.gen_data()
    k = 2
    # 循环找到最优分类
    best_assenment = np.inf
    best_centroids = None
    best_label = None
    for i in range(10):
        centroids, label, sum_assenment = k_means_.k_means(data, k)
        if sum_assenment < best_assenment:
            best_assenment = sum_assenment
            best_centroids = centroids
            best_label = label
    # 可视化
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(data[:, 0], data[:, 1])
    ax2.scatter(data_0[:, 0], data_1[:, 1], c='b', marker='o')
    ax2.scatter(data_1[:, 0], data_1[:, 1], c='g', marker='o')
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='*', s=180)
    plt.show()
