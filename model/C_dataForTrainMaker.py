# @Time    : 2024/4/27 23:28
# @Author  : ZJH
# @FileName: C_dataForTrainMaker.py
# @Software: PyCharm

import os
import utils.utils as tool
import numpy as np
from sklearn.model_selection import train_test_split

# actions = ["侧平举", "推肩", "俯卧撑", "二头弯举", "开合跳", , "箭步蹲"]
actions = ["侧平举", "推肩", "俯卧撑", "二头弯举", "开合跳", "深蹲", "箭步蹲"]


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
                num = num + 1
                if (isInvert):
                    dataset_with_labels.append((flipped_data, lable))
                    num = num + 1
    print(data_dir)
    print(num)


def getDataByPosition(data_dir, dataset_with_labels, label, isInvert, position):
    num = 0
    start = 0
    end = 0
    if (position == 1):
        start = 1
        end = 120
    if (position == 2):
        start = 121
        end = 240
    if (position == 3):
        start = 241
        end = 360
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_number = filename.split('.')[0]
            try:
                file_number = int(file_number)
                if start <= file_number <= end:
                    file_path = os.path.join(data_dir, filename)
                    with open(file_path, 'r') as file:
                        sample_data = [[float(value) for value in line.strip().split()] for line in file]
                        sample_data = tool.min_max_normalize(sample_data)
                        flipped_data = np.flip(sample_data, axis=1)
                        dataset_with_labels.append((sample_data, label))
                        num += 1
                        if isInvert:
                            dataset_with_labels.append((flipped_data, label))
                            num += 1
            except ValueError:
                continue
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


# 获取全部数据集 归一化数据
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


# 没有归一化数据 区分position(1,2,3)
def getDataSpecialPosition(dataSetsPath, isInvert, position):
    dataset_with_labels = []
    lable = 0
    for action in actions:
        action_path = os.path.join(dataSetsPath, action)
        datasets = getDataByPosition(action_path, dataset_with_labels, lable, isInvert, position)
        lable = lable + 1
    X = [data for data, label in dataset_with_labels]
    Y = [label for data, label in dataset_with_labels]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# x,y=getDataSpecialPosition("D:\py_project\RfidSport\model\dataSets", isInvert=0,position=1)

def newGetDataFromDatasets(data_dir, dataset_with_labels, lable):
    num = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(f'{file_path}', 'r') as file:
                sample_data = [[float(value) for value in line.strip().split()] for line in file]
                try:
                    sample_data = tool.min_max_normalize(sample_data)
                except Exception as e:
                    print(f"在归一化数据时发生了未知异常{data_dir}: {e}")
                dataset_with_labels.append((sample_data, lable))
                num = num + 1
    print(data_dir)
    print(num)


def newGetDatasetWithLabels(dataSetsPath):
    dataset_with_labels = []
    lable = 0
    for action in actions:
        action_path = os.path.join(dataSetsPath, action)
        datasets = newGetDataFromDatasets(action_path, dataset_with_labels, lable)
        lable = lable + 1
    return dataset_with_labels


def newGetXandY(dataSetsPath):
    dataset_with_labels = newGetDatasetWithLabels(dataSetsPath)
    X = [data for data, label in dataset_with_labels]
    Y = [label for data, label in dataset_with_labels]
    for j in range(0, len(X)):
        new_x = np.empty_like(X[0])
        for i in range(8):
            new_x[2 * i] = X[j][i]
            new_x[2 * i + 1] = X[j][i + 8]
        X[j] = new_x
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# newGetXandY("D:\py_project\RfidSport\model\dataSetsFINAL")

def getSingleAction9case(data_dir):
    X = [[] for _ in range(9)]
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(f'{file_path}', 'r') as file:
                sample_data = [[float(value) for value in line.strip().split()] for line in file]
                sample_data = tool.min_max_normalize(sample_data)
                num = int(filename.split('.')[0])
                X[(num - 1) // 40].append(sample_data)
    return np.array(X)

#对每个动作的不同采集位置和人员依次进行等比例随机划分
def getData9case(data_dir):
    Lable = 0
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for action in actions:
        action_path = os.path.join(data_dir, action)
        X = getSingleAction9case(action_path)
        for i in range(9):
            eachX = X[i]
            np.random.shuffle(eachX)
            split_idx = int(len(eachX) * 0.7)

            train = eachX[:split_idx]
            for each in train:
                X_train.append(each)
                Y_train.append(Lable)

            test = eachX[split_idx:]
            for each in test:
                X_test.append(each)
                Y_test.append(Lable)

        Lable = Lable + 1

    for j in range(0, len(X_train)):
        new_x = np.empty_like(X_train[0])
        for i in range(8):
            new_x[2 * i] = X_train[j][i]
            new_x[2 * i + 1] = X_train[j][i + 8]
        X_train[j] = new_x
    for j in range(0, len(X_test)):
        new_x = np.empty_like(X_test[0])
        for i in range(8):
            new_x[2 * i] = X_test[j][i]
            new_x[2 * i + 1] = X_test[j][i + 8]
        X_test[j] = new_x
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test


#getData9case("D:\py_project\RfidSport\model\dataSetsFINAL")
