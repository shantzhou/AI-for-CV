#!/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

class NMS:
    def __init__(self, box, score):
        self.box = box
        self.score = score
        self.dest = []

    def IOU(self, i, j):
        return i / i + j

    def nms(self, N_t):
        while self.box:
            m = np.argmax(self.score)
            b_m = self.box.pop(m)
            self.score.pop(m)
            self.dest.append(b_m)

            for b_i in self.box:
                if self.IOU(b_i, b_m) >= 0.5:
                    self.score.pop(box.index(b_i))
                    self.box.remove(b_i)
        return self.dest


if __name__ == '__main__':
    box = [5, 8, 10, 3, 7, 12, 1]
    score = [0.98, 0.5, 0.96, 0.21, 0.35, 0.68, 0.11]

    nms_ = NMS(box, score)
    print(nms_.nms(0.5))