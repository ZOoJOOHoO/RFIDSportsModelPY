# @Time    : 2024/4/16 16:48
# @Author  : ZJH
# @FileName: demo.py
# @Software: PyCharm
import copy
import utils.utils as tool
import utils.draw as darw
import math

phases = [[] for _ in range(8)]
rssis = [[] for _ in range(8)]

phase_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
rssi_numbers = [2, 4, 6, 8, 10, 12, 14, 16]

index = []
with open('D:/JAVAPROJECT/rfidSport/datas/二头弯举_done/3.txt', 'r') as file:
    for line_num, line in enumerate(file, 1):
        if line_num in phase_numbers:
            values = [float(value) for value in line.split()]
            phases[phase_numbers.index(line_num)] = values
        if line_num in rssi_numbers:
            values = [float(value) for value in line.split()]
            rssis[rssi_numbers.index(line_num)] = values
#darw.showData(2,4,0,2*math.pi,phases)
phases_raw = copy.deepcopy(phases)
for i in range(0, len(phases_raw)):
    tool.phaseJump(phases_raw[i])
    tool.phaseUnwrapping(phases_raw[i])
    phases_raw[i] = tool.moving_average_filter(phases_raw[i], 3)

darw.showData(2, 4, -1, 2 * math.pi+3, phases_raw)


for i in range(0, len(phases_raw)):
    start, end = tool.featureExtraction(phases_raw[i])
    index.append([start, end])
darw.showFeatureIndex1(2, 4, -1, 2 * math.pi+0.3, index, phases_raw)
