# @Time    : 2024/6/15 20:49
# @Author  : ZJH
# @FileName: loadDiffMoel.py
# @Software: PyCharm

import model.C_dataForTrainMaker as loadData
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

X_train, X_test, Y_train, Y_test=loadData.getData9case("D:\py_project\RfidSport\model\dataSetsFINAL")
model = load_model("demo_model")
loss, accuracy = model.evaluate(X_test, Y_test)
print(111)