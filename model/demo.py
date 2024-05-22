# @Time    : 2024/4/24 20:33
# @Author  : ZJH
# @FileName: demo.txt.py
# @Software: PyCharm
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import utils.utils as tool
import C_dataForTrainMaker
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import model.C_dataForTrainMaker as loadData
from tensorflow.keras.callbacks import LearningRateScheduler

# 加载数据
X, Y = loadData.newGetXandY("D:\py_project\RfidSport\model\dataSetsFINAL")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 对train数据添加反转 丰富数据
flipped_X_train = np.flip(X_train, axis=2)
X_train = np.concatenate((X_train, flipped_X_train), axis=0)
Y_train = np.concatenate((Y_train, Y_train), axis=0)


# flipped_X_test = np.flip(X_test, axis=2)
# X_test = np.concatenate((X_test, flipped_X_test), axis=0)
# Y_test = np.concatenate((Y_test, Y_test), axis=0)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((90, 16), input_shape=(16, 90)),
    tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation='softmax')
])

optimizer = Adam(lr=0.003)

# 使用自定义的 Adam 优化器实例来编译模型
model.compile(optimizer=optimizer,  # 使用自定义的 Adam 优化器
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # 评估指标


def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9  # 每20个epoch，将学习率乘以0.9
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)
model.fit(X_train, Y_train, epochs=500, batch_size=1024, validation_data=(X_test, Y_test), callbacks=[lr_scheduler])
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
