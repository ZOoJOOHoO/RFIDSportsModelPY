# @Time    : 2024/4/24 20:33
# @Author  : ZJH
# @FileName: demo.py
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


X_train, X_test, Y_train, Y_test = loadData.getXandY("D:\py_project\RfidSport\model\dataSets", isInvert=0)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((90, 16), input_shape=(16, 90)),
    tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=l2(0.01)),

    tf.keras.layers.Dense(8, activation='softmax')
])

optimizer = Adam(lr=0.002)

# 使用自定义的 Adam 优化器实例来编译模型
model.compile(optimizer=optimizer,  # 使用自定义的 Adam 优化器
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # 评估指标

# 训练模型
model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test))

test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print("Test Accuracy:", test_accuracy)

# 使用 model.predict() 获取预测概率，并使用 np.argmax() 获取预测的类别
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test_encoded, y_pred)
print("Confusion Matrix:")
print(cm)

# dataset_with_labels_for_validata = C_dataForTrainMaker.getDatasetWithLabels("D:\py_project\RfidSport\model\dataSetsdemo",isInvert=0)
# X_val = [data for data, label in dataset_with_labels_for_validata]
# y_val = [label for data, label in dataset_with_labels_for_validata]
#
# X_new = np.array(X_val)
# y_pred_probs1 = model.predict(X_new)
# y_pred1 = np.argmax(y_pred_probs1, axis=1)
# total = 0
# for y in y_pred1:
#     if (y == 0):
#         total = total + 1
# print(total / len(y_pred1))
