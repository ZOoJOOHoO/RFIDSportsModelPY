# @Time    : 2024/5/8 21:10
# @Author  : ZJH
# @FileName: draw_1_phaseUnwarpping.py
# @Software: PyCharm

from utils import singleTxtRead
import copy
import utils.reduceNoise as reduceNoise
import utils.utils as tool
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from matplotlib.font_manager import FontProperties

def showData(row, column, miny, maxy, datas):
    i = 1
    plt.figure(figsize=(16, 9))
    for data in datas:
        plt.subplot(row, column, i)
        if(i==1):
            plt.plot(data, color='b', label='Raw Data')
        elif(i==2):
            plt.plot(data, color='b', label='Unwrapped Data')
        else:
            plt.plot(data, color='b', label='Denoised Data')
        plt.ylim(miny, maxy)
        plt.xlim(0, len(data) - 1)
        plt.gca().tick_params(axis='both', length=4, direction='in', labelsize=15)
        plt.xticks(fontname='Times New Roman')  # 设置刻度数字字体为新罗马字体
        plt.yticks(fontname='Times New Roman')  # 设置刻度数字字体为新罗马字体
        ylabel = plt.ylabel('Phase (rad)', fontsize=20, fontname='Times New Roman')
        xlabel = plt.xlabel('Sample', fontsize=20, fontname='Times New Roman')
        plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 20}, handlelength=0.5,handletextpad=0.3)
        i = i + 1
    plt.tight_layout()
    #plt.savefig("相位解缠.svg", format='svg')
    plt.show()

phases, rssis = singleTxtRead.getData(r'D:\JAVAPROJECT\rfidSport\newData\positionA\深蹲\A\26.txt')
showDataList = []
phases_raw = copy.deepcopy(phases)
for i in range(0, len(phases_raw)):
    tool.phaseJump(phases_raw[i])
    tool.phaseUnwrapping(phases_raw[i])
showDataList.append(phases[4])
B= phases_raw[4]
showDataList.append(B)
for i in range(0, len(phases_raw)):
    phases_raw[i]=reduceNoise.wavelet_denoise2(phases_raw[i])
showDataList.append(phases_raw[4])

showData(1, 3, -0.5, 6.5, showDataList)