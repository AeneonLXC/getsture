# -*- ecoding: utf-8 -*-
# @ModuleName: alexNet
# @Function: AlexNet
# @Author: MysteriousKnight
# @Email: xingchenziyi@163.com
# @Time: 2021-09-30 14:14
from tensorflow.keras import layers, models, Model, Sequential
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
def AlexNet_v1(im_height=224, im_width=224, num_classes=6):
    """
    神经网络的模型结构
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

#  导入已经保存好的特征、标签数组
train_features = np.load("./img_binary/train_features.npy", allow_pickle=True)
train_labels = np.load("./img_binary/train_labels.npy", allow_pickle=True)

train_features = tf.convert_to_tensor(train_features / 255)

#  给标签进行独热码编码
train_labels = tf.squeeze(train_labels)
train_labels = tf.one_hot(train_labels, depth=6)

#  打印模型结构
mode.summary()

#  选择合适的优化器，loss函数，设置学习率
mode.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['categorical_crossentropy']
)

#  模型的保存路径
check = "./weight/gesture.ckpt"
if os.path.exists(check + ".index"):
    print("yes!!!!")
    mode.load_weights(check)
    
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    check, 
    save_weights_only=(True),
    save_best_only=(True)
    )

#  训练模型 可修改batch、epoch
history = mode.fit(
    train_features,
    train_labels,
    batch_size=256,
    epochs=100,
    validation_split=0.2,
    callbacks=([cp_callback])
)

#  获取历史的模型精度、loss信息
acc = history.history['categorical_crossentropy']
val_acc = history.history['val_categorical_crossentropy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#  绘制acc图像
plt.plot(acc, lw=3, label='acc')
plt.plot(val_acc, lw=3, label='val_acc')
plt.legend()
plt.show()

#  绘制loss图像
plt.plot(loss, lw=3, label='loss')
plt.plot(val_loss, lw=3, label='val_loss')
plt.legend()
plt.show()