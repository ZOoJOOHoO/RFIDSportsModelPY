# @Time    : 2024/6/4 20:45
# @Author  : ZJH
# @FileName: CNN.py
# @Software: PyCharm

import os
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
import model.C_dataForTrainMaker as loadData
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Flatten, Conv1D, Conv2D, \
    BatchNormalization, Reshape, GlobalAvgPool1D, ReLU, Add,GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

X_train, X_test, Y_train, Y_test=loadData.getData9case("D:\py_project\RfidSport\model\dataSetsFINAL")
indices = np.random.permutation(X_train.shape[0])
X_train = X_train[indices]
Y_train = Y_train[indices]
# # 加载数据
# X, Y = loadData.newGetXandY()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# # 对train数据添加反转 丰富数据
flipped_X_train = np.flip(X_train, axis=2)
X_train = np.concatenate((X_train, flipped_X_train), axis=0)
Y_train = np.concatenate((Y_train, Y_train), axis=0)

# flipped_X_test = np.flip(X_test, axis=2)
# X_test = np.concatenate((X_test, flipped_X_test), axis=0)
# Y_test = np.concatenate((Y_test, Y_test), axis=0)
all_kernel_size=3

def residual_block(x, filters, kernel_size=all_kernel_size, strides=1):
    shortcut = x
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x


def build_model():
    # 输入层
    inputs = Input(shape=(16, 90))
    x = Reshape((90, 16))(inputs)
    X = Conv1D(64, kernel_size=all_kernel_size, strides=1, padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    X = Conv1D(64, kernel_size=all_kernel_size, strides=1, padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = LSTM(units=256, input_shape=(90,16), return_sequences=True)(x)
    x = BatchNormalization()(x)
    tf.print(x.shape)
    x = Flatten()(x)
    tf.print(x.shape)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Dense(7, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()
optimizer = Adam(lr=0.006)

# 使用自定义的 Adam 优化器实例来编译模型
model.compile(optimizer=optimizer,  # 使用自定义的 Adam 优化器
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # 评估指标


def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 1  # 每20个epoch，将学习率乘以0.9
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)
history = model.fit(X_train, Y_train, epochs=300, batch_size=512, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
model.save("demo_model")
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
lossList = history.history['val_loss']
accList = history.history['val_accuracy']

with open('output.txt', 'w') as file:
    # 使用列表推导式将浮点数转换为字符串，并使用空格连接，然后写入第一行
    file.write(' '.join([str(loss) for loss in lossList]) + '\n')
    file.write(' '.join([str(acc) for acc in accList]) + '\n')

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(Y_test, predicted_classes)
row_sums = cm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_percentage = np.round(cm / row_sums * 100, decimals=2)
print("Percentage Confusion Matrix:")
print(cm_percentage)
