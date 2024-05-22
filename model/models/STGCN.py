import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import datetime
from sklearn.model_selection import KFold
import os
import model.C_dataForTrainMaker as loadData
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler

root = os.getcwd()
tf.config.experimental_run_functions_eagerly(True)

# 移除前部分的残差网络，定义新的图卷积层， 无池化，phi = relu(wx), 输出采用原始图结构，不再用残差连接, 全局分支加入卷积层

time_length = 90
tag_num = 8
channel = 2
d = 16


class GCNlayer(tf.keras.layers.Layer):
    def __init__(self, units=64):
        super(GCNlayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.X_w = self.add_weight(shape=(input_shape[1], input_shape[2]), initializer='glorot_uniform', trainable=True)
        self.G_w = self.add_weight(shape=(input_shape[2], self.units), initializer='glorot_uniform', trainable=True)

    def call(self, input):
        output = tf.multiply(input, self.X_w)
        re = tf.nn.relu(output)
        A = tf.matmul(re, re, transpose_b=True)
        time_ones = tf.Variable(lambda: tf.eye(input.shape[1]), trainable=False)
        time_ones = tf.reshape(time_ones, (1, input.shape[1], input.shape[1]))
        A_wavy = A + time_ones
        D = tf.linalg.diag(tf.math.pow(tf.reduce_sum(A_wavy, 2), -0.5))
        Z = tf.matmul(D, A_wavy)
        Z = tf.matmul(Z, D)
        Z = tf.matmul(Z, input)
        Z = tf.matmul(Z, self.G_w)
        return Z


def STGCN_model(s, t, num_classes):
    inputs_shape = tf.keras.Input(shape=(tag_num, time_length, channel), name="feature")

    # 时间分支
    time_x = tf.transpose(inputs_shape, perm=[0, 2, 3, 1])
    time_x = tf.keras.layers.Reshape((time_x.shape[1], time_x.shape[2] * time_x.shape[3]))(time_x)
    for i in range(t):
        time_x = GCNlayer()(time_x)
    time_out = tf.keras.layers.Flatten()(time_x)

    # 空间分支
    tag_x = tf.transpose(inputs_shape, perm=[0, 1, 3, 2])
    tag_x = tf.keras.layers.Reshape((tag_x.shape[1], tag_x.shape[2] * tag_x.shape[3]))(tag_x)
    for i in range(s):
        tag_x = GCNlayer()(tag_x)
    tag_out = tf.keras.layers.Flatten()(tag_x)

    # 全局分支的操作
    global_out = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs_shape)
    global_out = tf.keras.layers.Flatten()(global_out)

    # 合并三条分支
    concatenate_all = tf.keras.layers.concatenate([time_out, tag_out, global_out])
    # concatenate_all = tf.keras.layers.BatchNormalization()(concatenate_all)
    x = tf.keras.layers.Dense(256, activation="relu")(concatenate_all)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    STGCN_model = tf.keras.Model(inputs=inputs_shape, outputs=outputs, name="STGCN")
    STGCN_model.summary()
    # tf.keras.utils.plot_model(STGCN_model, to_file=root + "\\Spatial-Temporal_Graph_Convolutional_Network\\model_pic\\STGCN_s"+str(s)+"_t"+str(t)+".png", show_shapes=True)
    optimizer = Adam(lr=0.005)
    STGCN_model.compile(optimizer=optimizer,  # 使用自定义的 Adam 优化器
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # 评估指标
    return STGCN_model


def lr_schedule(epoch, lr):
    if epoch % 20 == 0 and epoch != 0:
        lr = lr * 0.95 # 每20个epoch，将学习率乘以0.9
    return lr



def Main():
    class_num = 7
    X, Y = loadData.newGetXandY("D:\py_project\RfidSport\model\dataSetsFINAL")
    X = np.array(X).reshape(X.shape[0], 8, 2, 90)
    X= X.transpose(0, 1, 3, 2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # 对train数据添加反转 丰富数据
    flipped_X_train = np.flip(X_train, axis=3)
    X_train = np.concatenate((X_train, flipped_X_train), axis=0)
    Y_train = np.concatenate((Y_train, Y_train), axis=0)

    # flipped_X_test = np.flip(X_test, axis=2)
    # X_test = np.concatenate((X_test, flipped_X_test), axis=0)
    # Y_test = np.concatenate((Y_test, Y_test), axis=0)



    #  data shape (1082,2,16,90)
    sgcn_num = 3
    tgcn_num = 3

    lr_scheduler = LearningRateScheduler(lr_schedule)
    model = STGCN_model(2,2, num_classes=class_num)
    history = model.fit(X_train, Y_train, epochs=500, batch_size=512, validation_data=(X_test, Y_test),
                        callbacks=[lr_scheduler])


if __name__ == "__main__":
    Main()

