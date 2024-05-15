# @Time    : 2024/4/22 12:43
# @Author  : ZJH
# @FileName: showFeatureIndex.py
# @Software: PyCharm

import copy
import utils.utils as tool
import utils.draw as darw
import math

phases = [[] for _ in range(8)]
rssis = [[] for _ in range(8)]

phase_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
rssi_numbers = [2, 4, 6, 8, 10, 12, 14, 16]


with open(r'D:\JAVAPROJECT\rfidSport\newData\positionB\深蹲\A\27.txt', 'r') as file:
    for line_num, line in enumerate(file, 1):
        if line_num in phase_numbers:
            values = [float(value) for value in line.split()]
            phases[phase_numbers.index(line_num)] = values
        if line_num in rssi_numbers:
            values = [float(value) for value in line.split()]
            rssis[rssi_numbers.index(line_num)] = values
index = []
phases_raw = copy.deepcopy(phases)
for i in range(0, len(phases_raw)):
    tool.phaseJump(phases_raw[i])
    tool.phaseUnwrapping(phases_raw[i])
    #phases_raw[i]=tool.reduceNoise(phases_raw[i])
    phases_raw[i] = tool.moving_average_filter(phases_raw[i], 3)
#darw.showData(2, 4, -2, 2 * math.pi+3, phases_raw)
for i in range(0, len(phases_raw)):
    start, end = tool.featureExtraction(phases_raw[i])
    index.append([start, end])
darw.showFeatureIndex1(2, 4, -1, 2 * math.pi+0.3, index, phases_raw)