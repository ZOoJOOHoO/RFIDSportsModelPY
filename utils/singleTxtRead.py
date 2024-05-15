# @Time    : 2024/5/8 0:30
# @Author  : ZJH
# @FileName: singleTxtRead.py
# @Software: PyCharm


def get():
    phases = [[] for _ in range(8)]
    rssis = [[] for _ in range(8)]
    phase_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
    rssi_numbers = [2, 4, 6, 8, 10, 12, 14, 16]
    return phases, rssis, phase_numbers, rssi_numbers

def getData(filePath):
    phases, rssis, phase_numbers, rssi_numbers = get()
    index = []
    with open(filePath, 'r') as file:
        for line_num, line in enumerate(file, 1):
            if line_num in phase_numbers:
                values = [float(value) for value in line.split()]
                phases[phase_numbers.index(line_num)] = values
            if line_num in rssi_numbers:
                values = [float(value) for value in line.split()]
                rssis[rssi_numbers.index(line_num)] = values
    return phases, rssis
