# @Time    : 2024/4/28 18:34
# @Author  : ZJH
# @FileName: A_cleanData.py
# @Software: PyCharm
import os

#dataSets重置为空
def clear_txt_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            folder_path = os.path.join(dirpath, dirname)
            clear_txt_files_in_folder(folder_path)

def clear_txt_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.txt'):
            os.remove(file_path)


root_dir = r'D:\py_project\RfidSport\model\dataSetsFINAL'
clear_txt_files(root_dir)
print("所有txt文件已删除")