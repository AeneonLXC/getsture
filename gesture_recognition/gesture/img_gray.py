# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 03:57:40 2021

@author: MysteriousKnight
@email: xingchenziyi@163.com
"""

import os
import numpy as np
import cv2 as cv

value = str(5)

for _ in range(int(value) + 1):
    path = "./img_augment/" + str(_) #数据集目录set
    img_path = os.listdir(path)
    for i in range(len(img_path)):
        img_np = cv.imread(path + "/" + img_path[i])
        #  将增广数据集中的图片转化为灰度图
        img_gray = cv.cvtColor(img_np,cv.COLOR_BGR2GRAY)
        #  将灰度图进行二值化处理
        mask = cv.inRange(img_gray, 20, 100)
        #  保存二值化图像
        cv.imwrite("./img_binary/" + str(_) + "/" + str(img_path[i]),mask)