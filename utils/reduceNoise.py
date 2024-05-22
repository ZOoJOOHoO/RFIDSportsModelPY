# @Time    : 2024/5/8 0:08
# @Author  : ZJH
# @FileName: reduceNoise.py
# @Software: PyCharm
import pywt
import numpy as np

import numpy as np
import pywt

def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
    # 如果信号长度为奇数，进行截断使其长度变为偶数
    if len(signal) % 2 == 1:
        signal = signal[:-1]

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 计算每个分解系数的阈值
    thresholds = []
    for i in range(1, len(coeffs)):
        sigma = np.median(np.abs(coeffs[i])) / 0.6745  # 估计标准差
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        thresholds.append(threshold)

    # 根据阈值对每个分解系数进行处理
    new_coeffs = []
    for i in range(len(coeffs) - 1):
        if threshold_type == 'hard':
            new_coeffs.append(pywt.threshold(coeffs[i + 1], thresholds[i], mode='hard'))
        elif threshold_type == 'soft':
            new_coeffs.append(pywt.threshold(coeffs[i + 1], thresholds[i], mode='soft'))
        else:
            raise ValueError("Invalid threshold type. Choose 'hard' or 'soft'.")

    # 重构去噪后的信号
    denoised_signal = pywt.waverec([coeffs[0]] + new_coeffs, wavelet)
    return denoised_signal
