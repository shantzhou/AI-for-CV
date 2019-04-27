#!/user/bin/env python3
# -*- coding:utf-8 -*-

'''
三层神经网络
'''
import numpy as np
import sklearn.datasets
import sklearn.linear_model

#生成数据集
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.2)

num_example = len(X)
nn_input_dim = 2
nn_output_dim = 2

lr = 0.01
reg_lambda = 0.01

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2


    #softmax
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    log_probs = -np.log(probs[range(num_example), y])
    loss = np.sum(log_probs)

    return 1./num_example * loss

def bulid_model(nn_hdim, num_passes=30000, print_loss = True):
    W1 = np.random.rand(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.rand(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    #Gradient descent
    for i in range(num_passes):
        #正向推导
        z1 = X.dot(W1) + b1  #(200, 10)
        a1 = np.tanh(z1)   #（200， 10）
        z2 = a1.dot(W2) + b2  #（200， 2）
        exp_scores = np.exp(z2)  #（200， 2）
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   #(200, 2)，各类别概率值

        #bp反向传播

        #求导
        delta3 = probs
        delta3[range(num_example), y] -= 1  #将softmax的结果Pi作为loss=cross_entropy中的Pi,求导并将结果传回

        dW2 = (a1.T).dot(delta3)  #(10, 2)
        db2 = np.sum(delta3, axis=0, keepdims=True)  #(1, 2)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))  #(200, 10)  tanh derivative
        dW1 = (X.T).dot(delta2) #(10, 10)
        db1 = np.sum(delta2, axis=0) #(1, 10)

        #optional
        W1 += -lr * dW1
        b1 += -lr * db1
        W2 += -lr * dW2
        b2 += -lr * db2

        model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

        if print_loss and i%1000==0:
            print('loss after iteration %i:%f'%(i, calculate_loss(model)))
    return model

# n-dimesional hidden layer表示hidden里有10个神经元
model = bulid_model(10, print_loss=True)



