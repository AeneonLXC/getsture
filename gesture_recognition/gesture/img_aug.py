# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 03:11:06 2021

@author: MysteriousKnight
@email: xingchenziyi@163.com
"""

import cv2 as cv
import numpy as np
import os

img = cv.imread("./img_main/0/" + "0_1.jpg")
value = str(5)

rows, cols = img.shape[:2] # 获取图片的长cols和宽rows
echo = 5
for e in range(echo):
    M = np.float32([[1, 0, -30 + e * 20], [0, 1, -40]])
    for _ in range(int(value) + 1):
        path = "./img_main/" + str(_) #数据集目录set
        img_path = os.listdir(path) # 获取到img的路径
        for i in range(len(img_path)):
            img_np = cv.imread(path + "/" + img_path[i])
            
            # img1 = cv.flip(img_np, -90) # 图像翻转增广
            dst = cv.warpAffine(img_np, M, (cols, rows)) # 图像平移增广
            cv.imshow("imn", img_np)
            cv.imwrite("./img_augment/" + str(_)+ "/dst_" + str(e) + "_" + str(img_path[i]), dst)

            cul = cv.flip(img_np, -90)# 图像翻转增广
            cv.imwrite("./img_augment/" + str(_)+ "/cul_" + str(e) + "_" + str(img_path[i]), cul)
            
            mat_rotate = cv.getRotationMatrix2D((rows * 0.3, cols * 0.3), 45 + e * 6, 1)  # 图像旋转增广
            dst1 = cv.warpAffine(img_np, mat_rotate, (rows, cols))
            cv.imwrite("./img_augment/" + str(_)+ "/dst1_" + str(e) + "_" + str(img_path[i]), dst1)
            
            mat_rotate = cv.getRotationMatrix2D((rows * 0.3, cols * 0.3), - e * 6 - 15, 1)  # 图像旋转增广
            dst2 = cv.warpAffine(img_np, mat_rotate, (rows, cols))
            cv.imwrite("./img_augment/" + str(_)+ "/dst2_" + str(e) + "_" + str(img_path[i]), dst2)