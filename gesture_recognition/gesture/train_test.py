# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 04:24:40 2021

@author: MysteriousKnight
@email: xingchenziyi@163.com
"""

import os
import numpy as np
import cv2 as cv

value = str(5)
train = []

for _ in range(int(value) + 1):
    path = "./img_binary/" + str(_) #数据集目录set
    img_path = os.listdir(path)
    for i in range(len(img_path)):
        img_np = cv.imread(path + "/" + img_path[i])
        # 将img_binary中的图片导入tiran中
        train.append(img_np)

#  创建标签数组
total_feature = np.array(train) 
fig_0 = [0 for i in range(len(img_path))]
fig_1 = [1 for i in range(len(img_path))]
fig_2 = [2 for i in range(len(img_path))]
fig_3 = [3 for i in range(len(img_path))]
fig_4 = [4 for i in range(len(img_path))]
fig_5 = [5 for i in range(len(img_path))]

total_label = fig_0 + fig_1 + fig_2 + fig_3 + fig_4 + fig_5
total_label = np.array(total_label)

#  打乱特征、标签，一一对应
np.random.seed(1)
index = np.random.permutation(total_label.size)

train_features = total_feature[index] # 此时数组已经打乱
train_labels = total_label[index] # 此时数组已经打乱

#  保存特征数组和标签数组
np.save("./img_binary/train_features.npy", train_features)
np.save("./img_binary/train_labels.npy", train_labels)