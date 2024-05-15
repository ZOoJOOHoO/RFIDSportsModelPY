import copy
import math
import os
import utils.utils as tool
import utils.draw as darw

# 调用函数并指定文件夹路径
directory_path = r"D:\QQ\文件\ZJH_下蹲"
directory_path2 = r"D:\QQ\文件\ZJH_弯举"

datas = []
for i in range(1, 10):
    path = os.path.join(directory_path2, f"{i}.txt")
    datas.append(tool.readRawData(path)[0])

showData = []
indexs = []
start, end = 0, 0
for i in range(0, len(datas)):
    datas[i] = [float(item) for item in datas[i]]
    unwrapping = tool.phaseUnwrapping(datas[i][0:300])
    average_filter = tool.moving_average_filter(unwrapping, 7)
    showData.append(average_filter)
    start, end = tool.featureExtraction(average_filter, 100)
    indexs.append([start, end])

darw.showFeatureIndex(3, 3, -2, 8, indexs, showData)