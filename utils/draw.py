# @Time    : 2024/4/2 20:18
# @Author  : ZJH
# @FileName: draw.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 画图
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


# 特征部分数据展示
def showFeatureIndex(row, column, miny, maxy, indexs, datas):
    i = 1
    for index, data in zip(indexs, datas):
        start = index[0]
        end = index[1]
        plt.subplot(row, column, i)
        i = i + 1
        plt.plot(data, color='b')
        plt.ylim(miny, maxy)
        plt.axvline(x=start, color='r', linestyle='--')
        plt.axvline(x=end, color='r', linestyle='--')
    plt.show()


def showFeatureIndex1(row, column, miny, maxy, indexs, datas):
    i = 1
    plt.figure(figsize=(12, 6))
    for index, data in zip(indexs, datas):
        start = index[0]
        end = index[1]
        plt.subplot(row, column, i)
        i += 1
        plt.plot(data, color='orange')

        # 计算索引部分的最大值和最小值
        max_part = max(data[start:end + 1])+0.2
        min_part = min(data[start:end + 1])-0.2


        # 绘制包围索引部分的矩形
        rect = Rectangle((start, min_part), end - start, max_part - min_part, linewidth=1, edgecolor='r',
                         facecolor='none')
        plt.gca().add_patch(rect)

        # 调整y轴范围
        plt.ylim(miny, maxy)
        plt.savefig("特征提取.svg", format='svg')
    plt.show()
