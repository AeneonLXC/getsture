# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:35:31 2021

@author: MysteriousKnight
@email: xingchenziyi@163.com
"""

import os
import numpy as np

value = str(5)

for _ in range(int(value) + 1):
    path = "./img_augment/" + str(_) #数据集目录set
    img_path = os.listdir(path)
    for i in range(len(img_path)):
        try:
            os.rename(path + "/" + str(img_path[i]), path + "/"  + str(_)  + "_" + str(i + 1) + ".jpg")
        except:
            pass
