# @Time    : 2024/5/9 19:28
# @Author  : ZJH
# @FileName: draw_2_featureIndexShow.py
# @Software: PyCharm

from utils import singleTxtRead
import copy
import utils.reduceNoise as reduceNoise
import utils.utils as tool
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from matplotlib.font_manager import FontProperties

def showFeatureIndex1(row, column, miny, maxy, indexs, datas):
    i = 1
    plt.figure(figsize=(16, 6))
    for index, data in zip(indexs, datas):
        start = index[0]
        end = index[1]
        plt.subplot(row, column, i)
        if i == 1:
            plt.plot(data, color='b', label='Tag 1')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 2:
            plt.plot(data, color='b', label='Tag 2')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 3:
            plt.plot(data, color='b', label='Tag 3')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 4:
            plt.plot(data, color='b', label='Tag 4')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 5:
            plt.plot(data, color='b', label='Tag 5')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 6:
            plt.plot(data, color='b', label='Tag 6')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 7:
            plt.plot(data, color='b', label='Tag 7')
            plt.ylim(min(data)-1, max(data)+1)
        elif i == 8:
            plt.plot(data, color='b', label='Tag 8')
            plt.ylim(min(data)-1, max(data)+1)

        # 计算索引部分的最大值和最小值
        max_part = max(data[start:end + 1])+0.2
        min_part = min(data[start:end + 1])-0.2
        plt.xlim(0, len(data) - 1)
        plt.gca().tick_params(axis='both', length=4, direction='in', labelsize=15)
        plt.xticks(fontname='Times New Roman')  # 设置刻度数字字体为新罗马字体
        plt.yticks(fontname='Times New Roman')  # 设置刻度数字字体为新罗马字体
        ylabel = plt.ylabel('Phase (rad)', fontsize=20, fontname='Times New Roman')
        xlabel = plt.xlabel('Sample', fontsize=20, fontname='Times New Roman')
        plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 20}, handlelength=0.5,handletextpad=0.3)

        # 绘制包围索引部分的矩形
        rect = Rectangle((start, min_part), end - start, max_part - min_part, linewidth=1.5, edgecolor='r',
                         facecolor='none', linestyle='--', alpha=0.5)  # 设置线型、线宽、透明度等
        plt.gca().add_patch(rect)

        # 调整y轴范围
        #plt.ylim(miny, maxy)

        i += 1
    plt.tight_layout()
    #plt.savefig("特征提取.svg", format='svg')
    plt.show()

# phases, rssis = singleTxtRead.getData(r'D:\JAVAPROJECT\rfidSport\newData\positionA\深蹲\A\5.txt')
phases, rssis = singleTxtRead.getData(r'D:\JAVAPROJECT\rfidSport\newData\positionA\深蹲\A\40.txt')
phases_raw = copy.deepcopy(phases)
for i in range(0, len(phases_raw)):
    tool.phaseJump(phases_raw[i])
    tool.phaseUnwrapping(phases_raw[i])
    phases_raw[i]= reduceNoise.wavelet_denoise(phases_raw[i])
index = []
for i in range(0, len(phases_raw)):
    start, end = tool.featureExtraction(phases_raw[i])
    index.append([start, end])
showFeatureIndex1(2, 4, 0, 2 * math.pi+1.5, index, phases_raw)