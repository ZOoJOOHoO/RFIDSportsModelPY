# @Time    : 2024/5/7 21:01
# @Author  : ZJH
# @FileName: resnet.py
# @Software: PyCharm

import model.C_dataForTrainMaker as loadData
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(88)
np.random.seed(88)
dataset_with_labels = loadData.getDatasetWithLabels("D:\py_project\RfidSport\model\dataSets", isInvert=1)
X = [data for data, label in dataset_with_labels]
XX=[]
for i in range(len(X)):#将（8,220）转置为（220,8）  220为时间步长 8为特征数（RSSI PHASE）
    XX.append(X[i].T)
XX=np.array(XX)
Y = [label for data, label in dataset_with_labels]
indices = np.random.permutation(len(XX))
X = [XX[i] for i in indices]
Y = [Y[i] for i in indices]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_train = np.array(X_train)
X_test = np.array(X_test)

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

class ResnetBlock(Model):  # 残差块
    def __init__(self, filters, filters_size, strides, residual_path=False):  # residual_path=False 通道数是否变化
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.filter_size = filters_size
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv1D(filters=filters, kernel_size=filters_size, strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.c2 = Conv1D(filters=filters, kernel_size=filters_size, strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()
        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same',
                                  use_bias=False)  # 1*1卷积 保证维度相同
            self.down_b1 = BatchNormalization()
        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        y = self.b2(x)
        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)
        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet(Model):
    def __init__(self):
        super(ResNet, self).__init__()
        size = 5
        self.c1 = Conv1D(filters=32, kernel_size=size, strides=2, padding='same', input_shape=(90, 16))
        self.c2 = Conv1D(filters=64, kernel_size=size, strides=2, padding='same')
        self.pool = MaxPooling1D(pool_size=3, strides=2)
        self.block1 = ResnetBlock(32, size, strides=2, residual_path=True)
        self.block2 = ResnetBlock(32, size, strides=1)
        self.block3 = ResnetBlock(32, size, strides=1)
        self.block4 = ResnetBlock(32, size, strides=1)

        self.block7 = ResnetBlock(64, size, strides=2, residual_path=True)
        self.block8 = ResnetBlock(64, size, strides=1)
        self.block9 = ResnetBlock(64, size, strides=1)
        self.block10 = ResnetBlock(64, size, strides=1)

        self.block13 = ResnetBlock(128, size, strides=2, residual_path=True)
        self.block14 = ResnetBlock(128, size, strides=1)
        self.block15 = ResnetBlock(128, size, strides=1)
        self.block16 = ResnetBlock(128, size, strides=1)

        self.pool_1 = GlobalAveragePooling1D()
        self.Dense1 = Dense(8, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        x = self.pool_1(x)
        y = self.Dense1(x)
        return y


def train():
    model = ResNet()
    optimizer = Adam(lr=0.002)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    model.fit(X_train, Y_train_encoded, batch_size=128, epochs=200, validation_data=(X_test, Y_test_encoded),
              validation_freq=1)

train()
