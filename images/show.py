# @Time    : 2024/5/7 19:10
# @Author  : ZJH
# @FileName: show.py
# @Software: PyCharm

from utils import singleTxtRead
import copy
import utils.reduceNoise as reduceNoise
import utils.utils as tool
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

def showData(row, column, miny, maxy, datas):
    i = 1
    plt.figure(figsize=(12, 6))
    for data in datas:
        plt.subplot(row, column, i)
        i = i + 1
        plt.plot(data, color='b')
        plt.ylim(miny, maxy)
        ylabel = plt.ylabel('相\n位', fontsize=14, rotation=0, multialignment='center',rotation_mode='anchor',va='center_baseline',labelpad=10)
    plt.tight_layout()
    #plt.savefig("降噪.svg", format='svg')
    plt.show()

phases, rssis = singleTxtRead.getData(r'D:\JAVAPROJECT\rfidSport\newData\positionA\深蹲\A\33.txt')

phases_raw = copy.deepcopy(phases)
for i in range(0, len(phases_raw)):
    tool.phaseJump(phases_raw[i])
    tool.phaseUnwrapping(phases_raw[i])
    #phases_raw[i] = tool.moving_average_filter(phases_raw[i], 3)
    #phases_raw[i]=tool.reduceNoise(phases_raw[i])
    #phases_raw[i]=reduceNoise.wavelet_denoise2(phases_raw[i])
showDataList = []
showDataList.append(phases_raw[7])
showData(2, 4, -2, 2 * math.pi + 3, phases)
