# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 02:05:37 2021

@MysteriousKnight: 23608
@Email: xingchenziyi@163.com
"""

from tensorflow.keras import layers, models, Model, Sequential
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
def AlexNet_v1(im_height=224, im_width=224, num_classes=6):
    """
    :param im_height:
    :param im_width:
    :param num_classes:
    :return:
    """
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # output(None, 227, 227, 3)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)  # output(None, 55, 55, 48)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 27, 27, 48)
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 6, 6, 128)
    x = layers.Flatten()(x)  # output(None, 6*6*128)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)  # output(None, 2048)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation="relu")(x)  # output(None, 2048)
    x = layers.Dense(num_classes)(x)  # output(None, 6)
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)
    return model

mode = AlexNet_v1()

# 调用已经训练好的模型
check = "./weight/gesture.ckpt"
mode.load_weights(check)

# 获取摄像头的接口
cap = cv.VideoCapture(0) # 0一般都是你电脑内置的摄像头 1是外置摄像头
w = 224
h = 224

while 1:# 1代表Ture 0代表False
    bool_value, frame = cap.read()
    #  将摄像头进行水平翻转
    frame = cv.flip(frame, 90)
    #  将获取到的视频流进行转灰度图
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #   将灰度图二值化处理
    mask = cv.inRange(gray, 20, 80)
    #   裁剪图像，归一化，方便喂入神经网络
    img = mask[0:w,0:h] / 255
    # img = img.astype(np.uint8)
    #   对图像进行魔改
    img3w = np.array([img.astype(np.uint8),img.astype(np.uint8),img.astype(np.uint8)])
    #   对图像的维度进行修改
    img3w = img3w.reshape([w,h,3])
    #   对图像多增加一个维度，因为喂入神经网络需要四个维度，第一个维度代表喂入几张图片
    img_np = np.expand_dims(img3w, axis=0)
    #   获取预测结果
    result = np.argmax(mode.predict(img_np))
    #   画框
    cv.rectangle(frame, (0,0), (w,h), (0,255,0))
    cv.imshow("frame", frame)
    cv.imshow("img", img)
    # cv.imshow("mask", img3w)
    print(result)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()