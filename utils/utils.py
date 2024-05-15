# @Time    : 2024/3/26 23:06
# @Author  : ZJH
# @FileName: utils.py
# @Software: PyCharm

import math
import matplotlib.pyplot as plt
import os
import pywt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 去除π值跳变
def phaseJump(data):
    minThreshold = math.pi - 0.6
    maxThreshold = math.pi + 0.6
    for i in range(1, len(data) - 1):
        Ldiff = data[i] - data[i - 1]
        Rdiff = data[i] - data[i + 1]
        if (abs(Ldiff) > minThreshold and abs(Ldiff < maxThreshold) and abs(Rdiff) > minThreshold and abs(
                Rdiff) < maxThreshold and Ldiff * Rdiff > 0):
            if (Ldiff > 0):
                data[i] = data[i] - math.pi
            else:
                data[i] = data[i] + math.pi
    if (abs(data[0] - data[1]) > minThreshold and abs(data[0] - data[1]) < maxThreshold):
        if (data[0] < data[1]):
            data[0] = data[0] + math.pi
        else:
            data[0] = data[0] - math.pi

# 解决相位缠绕
def phaseUnwrapping(data):
    sum = data[0]
    for i in range(1, len(data)):
        mean = sum / i
        diff = data[i] - mean
        if (diff > math.pi):
            data[i] -= 2 * math.pi
        if (diff < -1 * math.pi):
            data[i] += 2 * math.pi
        sum += data[i]
    return data


# 小波降噪
def reduceNoise(data):
    coeffs = pywt.wavedec(data, 'db1', level=5)  # 使用'db1'小波基，分解3层
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1]))  # 估计噪声的标准差
    threshold = sigma * np.sqrt(2 * np.log(len(data)))  # 设置阈值
    # 对小波系数进行阈值处理
    new_coeffs = coeffs.copy()
    new_coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    # 小波逆变换重构信号
    denoised_signal = pywt.waverec(new_coeffs, 'db1')
    return denoised_signal


def moving_average_filter(data, window_size):
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("窗口大小必须是正奇数。")
        # 使用列表推导式计算移动平均值
    filtered_data = [sum(data[i:i + window_size]) / window_size for i in range(len(data) - window_size + 1)]
    # 如果原始数据长度大于窗口大小，填充剩余部分的数据
    filtered_data += [data[-1]] * (len(data) - len(filtered_data))
    return filtered_data


# 读取数据
def readRawData(filePath):
    phase = []
    rssi = []
    if os.path.exists(filePath):
        try:
            with open(filePath, 'r') as file:
                for line in file:
                    data = line.split()
                    if len(data) >= 3:
                        phase.append(data[2])
                        rssi.append(data[1])
        except FileNotFoundError:
            print(f"Error: not found file")
    return phase, rssi


# 划分不同区间，以区间为单位计算各自方差，找到连续区间方差和最大的部分  size = 42
def featureExtraction(data):
    variances = []  # 存储每个窗口的方差
    window_size1 = 5
    step1 = 1
    window_size2 = 86
    max_sum = 0
    start, end = 0, 0
    for i in range(0, len(data) - window_size1 + 1, step1):
        window = data[i:i + window_size1]
        variance = np.var(window)
        variances.append(variance)
    for i in range(len(variances) - window_size2 + 1):
        window_sum = sum(variances[i:i + window_size2])
        if window_sum > max_sum:
            max_sum = window_sum
            start, end = i, i + window_size2 - 1
    return start * step1, end * step1 + window_size1 - 1

def write_lists_to_txt(lists_of_elements, filename):
    with open(filename, 'w') as file:
        for lst in lists_of_elements:
            # 将列表转换为字符串，元素之间用空格分隔
            line = ' '.join(map(str, lst))
            # 写入文件，每个列表占一行
            file.write(line + '\n')

def min_max_normalize(sample_data):
    # 沿着维度的方向找到最小值和最大值
    min_values = np.min(sample_data, axis=1, keepdims=True)
    max_values = np.max(sample_data, axis=1, keepdims=True)
    # 进行01标准化
    normalized_data = (sample_data - min_values) / (max_values - min_values)
    return normalized_data

