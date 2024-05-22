# @Time    : 2024/5/21 19:29
# @Author  : ZJH
# @FileName: lstm.py
# @Software: PyCharm
import model.C_dataForTrainMaker as loadData
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout

X_train, X_test, Y_train, Y_test = loadData.getXandY("D:\py_project\RfidSport\model\dataSets", isInvert=0)

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(16, 90), return_sequences=True))
model.add(Dropout(0.2))  # 添加Dropout层防止过拟合


model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))  # 添加Dropout层防止过拟合

# 添加一个全连接层
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  # 添加Dropout层防止过拟合

# 添加输出层，使用softmax激活函数进行多分类
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=200, batch_size=128, validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')