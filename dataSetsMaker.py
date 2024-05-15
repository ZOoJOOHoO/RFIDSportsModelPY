# @Time    : 2024/4/22 17:37
# @Author  : ZJH
# @FileName: dataSetsMaker.py
# @Software: PyCharm

import copy
import utils.utils as tool
import utils.draw as darw
import math

phases = [[] for _ in range(8)]
rssis = [[] for _ in range(8)]
phase_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
rssi_numbers = [2, 4, 6, 8, 10, 12, 14, 16]
for num in range(1, 700):
    index = []
    with open(f'D:/JAVAPROJECT/rfidSport/datas/箭步蹲L/{num}.txt', 'r') as file:
        for line_num, line in enumerate(file, 1):
            if line_num in phase_numbers:
                values = [float(value) for value in line.split()]
                phases[phase_numbers.index(line_num)] = values
            if line_num in rssi_numbers:
                values = [float(value) for value in line.split()]
                rssis[rssi_numbers.index(line_num)] = values
    phases_raw = copy.deepcopy(phases)
    flag = 0
    for i in range(0, len(phases_raw)):
        if (len(phases_raw[i]) < 42):
            print(num)
            flag=1
            break
        tool.phaseJump(phases_raw[i])
        tool.phaseUnwrapping(phases_raw[i])
        phases_raw[i] = tool.moving_average_filter(phases_raw[i], 5)
        start, end = tool.featureExtraction(phases_raw[i])
        index.append([start, end])
        phases_raw[i] = phases_raw[i][start:end + 1]
        rssis[i] = rssis[i][start:end + 1]

    # darw.showData(2, 4, -1, 2 * math.pi + 3, phases_raw)
    if(flag==0):
        tool.write_lists_to_txt(phases_raw + rssis, f'dataSets/下蹲_done/{num}.txt')
        #darw.showFeatureIndex1(2, 4, -1, 2 * math.pi + 0.3, index, phases_raw)
