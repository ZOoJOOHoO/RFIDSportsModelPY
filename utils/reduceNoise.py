# @Time    : 2024/5/8 0:08
# @Author  : ZJH
# @FileName: reduceNoise.py
# @Software: PyCharm
import pywt
import numpy as np

def wavelet_denoise(signal, wavelet='db4', level=2):
    # 将信号进行小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 对每个分解系数进行软阈值处理
    threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
    # 重构去噪后的信号
    denoised_signal = pywt.waverec([coeffs[0]] + new_coeffs, wavelet)
    return denoised_signal


def wavelet_denoise2(signal, wavelet='db4', level=3, threshold_type='soft'):
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