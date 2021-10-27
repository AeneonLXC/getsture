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
        # 转换为hsv
        img_hsv = cv.cvtColor(img_np,cv.COLOR_BGR2HSV)
        cv.imwrite("./img_hsv/" + str(_) + "/" + str(img_path[i]),img_hsv)