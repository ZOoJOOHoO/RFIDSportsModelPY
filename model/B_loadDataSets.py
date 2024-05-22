# @Time    : 2024/4/25 18:43
# @Author  : ZJH
# @FileName: B_loadDataSets.py
# @Software: PyCharm

import os
import copy
import utils.utils as tool
import utils.reduceNoise as reduceNoise
from utils.singleTxtRead import getData as getPhasesAndRssis

dataDir = r'D:\JAVAPROJECT\rfidSport\newData'
positions = ["positionA", "positionB", "positionC"]
actions = ["深蹲", "高抬腿", "箭步蹲" , "侧平举", "推肩", "俯卧撑", "二头弯举","开合跳"]
persions = ["A","zjs","pk"]

def loadData():
    for action in actions:
        num = 1
        for position in positions:
            for persion in persions:
                base_path = os.path.join(dataDir, position, action, persion)
                print(base_path)
                if os.path.exists(base_path) and os.path.isdir(base_path):
                    for root, dirs, files in os.walk(base_path):
                        print(len(files))
                        count = 1
                        for file in files:
                            if(count==41):
                                break
                            if file.endswith('.txt'):
                                count=count+1
                                file_path = os.path.join(root, file)
                                num = getPhaseandRssi(file_path, action, num)

def getPhaseandRssi(file_path, action, num):
    phases, rssis = getPhasesAndRssis(file_path)
    phases_raw = copy.deepcopy(phases)
    flag = 0
    for i in range(0, len(phases_raw)):
        if (len(phases_raw[i]) < 90):
            flag = 1
            print(file_path)
            break
        tool.phaseJump(phases_raw[i])
        tool.phaseUnwrapping(phases_raw[i])
        #phases_raw[i] = tool.moving_average_filter(phases_raw[i], 5)
        phases_raw[i] = reduceNoise.wavelet_denoise(phases_raw[i])
        start, end = tool.featureExtraction(phases_raw[i])
        rssis[i] = rssis[i][start:end+1]
        phases_raw[i] = phases_raw[i][start:end+1]
    if (flag == 0):
        #write_lists_to_txt(phases_raw + rssis, f'dataSetsFINAL/{action}/{num}.txt')
        num = num + 1
    return num


def write_lists_to_txt(lists_of_elements, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        for lst in lists_of_elements:
            line = ' '.join(map(str, lst))
            file.write(line + '\n')

#loadData()
