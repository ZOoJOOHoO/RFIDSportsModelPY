# @Time    : 2024/4/27 23:28
# @Author  : ZJH
# @FileName: C_dataForTrainMaker.py
# @Software: PyCharm

import os
import utils.utils as tool
import numpy as np
from sklearn.model_selection import train_test_split

actions = ["深蹲", "箭步蹲", "高抬腿", "二头弯举","开合跳","侧平举", "推肩", "俯卧撑"]  #

def getDataFromDatasets(data_dir, dataset_with_labels, lable, isInvert):
    num = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(f'{file_path}', 'r') as file:
                sample_data = [[float(value) for value in line.strip().split()] for line in file]
                try:
                    sample_data = tool.min_max_normalize(sample_data)
                    flipped_data = np.flip(sample_data, axis=1)
                except Exception as e:
                    print(f"在归一化数据时发生了未知异常{data_dir}: {e}")
                dataset_with_labels.append((sample_data, lable))
                if (isInvert):
                    dataset_with_labels.append((flipped_data, lable))
                    num = num + 2
                else:
                    dataset_with_labels.append((sample_data, lable))
                    num = num + 1
    print(data_dir)
    print(num)


def getDatasetWithLabels(dataSetsPath, isInvert):
    dataset_with_labels = []
    lable = 0
    for action in actions:
        action_path = os.path.join(dataSetsPath, action)
        datasets = getDataFromDatasets(action_path, dataset_with_labels, lable, isInvert)
        lable = lable + 1
    return dataset_with_labels


def getXandY(dataSetsPath, isInvert):
    dataset_with_labels = getDatasetWithLabels(dataSetsPath, isInvert)
    X = [data for data, label in dataset_with_labels]
    Y = [label for data, label in dataset_with_labels]
    X = np.array(X)
    Y = np.array(Y)
    np.random.seed(88)
    indices = np.random.permutation(len(X))
    X = [X[i] for i in indices]
    Y = [Y[i] for i in indices]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test
