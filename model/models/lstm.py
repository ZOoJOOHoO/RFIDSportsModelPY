# @Time    : 2024/5/21 19:29
# @Author  : ZJH
# @FileName: lstm.py
# @Software: PyCharm
import model.C_dataForTrainMaker as loadData
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = loadData.getXandY("D:\py_project\RfidSport\model\dataSets", isInvert=0)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(64, activation='relu')))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(8, activation='softmax'))


# 自定义学习率递减函数
def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9  # 每20个epoch，将学习率乘以0.9
    return lr


# 定义学习率递减回调函数
lr_scheduler = LearningRateScheduler(lr_schedule)

# 编译模型
optimizer = Adam(lr=0.002)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型，并将历史记录存储在history变量中
history = model.fit(X_train, Y_train, epochs=300, batch_size=128, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])

# 提取训练过程中的损失和准确率
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 计算并打印测试集上的损失和准确率
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
