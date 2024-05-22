import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Flatten, Conv1D,BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import model.C_dataForTrainMaker as loadData
import numpy as np
from sklearn.model_selection import train_test_split
import random

# 加载数据
X, Y = loadData.newGetXandY("D:\py_project\RfidSport\model\dataSetsFINAL")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#对train数据添加反转 丰富数据
flipped_X_train = np.flip(X_train, axis=2)
X_train = np.concatenate((X_train, flipped_X_train), axis=0)
Y_train = np.concatenate((Y_train, Y_train), axis=0)

# #对test数据添加反转
# flipped_X_test = np.flip(X_test, axis=2)
# X_test = np.concatenate((X_test, flipped_X_test), axis=0)
# Y_test = np.concatenate((Y_test, Y_test), axis=0)

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

def scaled_dot_product_attention(q, k, v):
    """
    缩放点积注意力
    """
    # Q K点积
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dk即为K的最后一个维度
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention_weights和V相乘，产生输出
    output = tf.matmul(attention_weights, v)

    return output, attention_weights
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):  # 16
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # 2

        # 分别定义Q K V的投影层
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # 定义最后的dense层
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        划分多头

        分拆最后一个维度d_model到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 对Q K V进行投影
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 对Q K V划分多头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 并行计算多个Q K V的缩放点积注意力
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # 通过reshape，将attention的结果拼接起来
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, 90, self.d_model))  # (batch_size, seq_len_q, d_model)

        # 将拼接后的结果输入全连接层，产生输出
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
num_heads = 4
d_model = 16


# 构建模型
def build_MHA_LSTM_model():
    # 输入层
    inputs = Input(shape=(16, 90))
    x = tf.keras.layers.Reshape((90, 16))(inputs)
    #x = MultiHeadAttention(d_model, num_heads)(x, x, x, mask=None)

    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)

    tf.print(x.shape)
    x =Flatten()(x)
    tf.print(x.shape)

    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Dense(8, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# 创建模型
model = build_MHA_LSTM_model()

# 编译模型
optimizer = Adam(lr=0.003)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 定义学习率递减函数
def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9# 每20个epoch，将学习率乘以0.9
    return lr


# 定义学习率递减回调函数
lr_scheduler = LearningRateScheduler(lr_schedule)

# 训练模型，并将历史记录存储在history变量中
history = model.fit(X_train, Y_train, epochs=500, batch_size=1024, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])

# 计算并打印测试集上的损失和准确率
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# history = model.fit(X_train, Y_train, epochs=300, batch_size=1024, callbacks=[lr_scheduler])
# test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
# print('Test loss:', test_loss)
# print('Test accuracy:', test_acc)


# model.save("MHA_LSTM_model")
# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs', fontsize=20, fontname='Times New Roman')
# plt.ylabel('Loss', fontsize=20, fontname='Times New Roman')
# plt.gca().tick_params(axis='both', length=4, direction='in', labelsize=15)
# plt.xticks(fontname='Times New Roman')  # 设置刻度数字字体为新罗马字体
# plt.yticks(fontname='Times New Roman')  # 设置刻度数字字体为新罗马字体
# plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 20}, handlelength=0.5, handletextpad=0.3)
# plt.savefig("损失曲线.svg", format='svg')
# plt.show()
#
# # 绘制准确率曲线
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs', fontsize=20, fontname='Times New Roman')
# plt.ylabel('Accuracy', fontsize=20, fontname='Times New Roman')
# plt.gca().tick_params(axis='both', length=4, direction='in', labelsize=15)
# plt.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 10}, handlelength=0.5, handletextpad=0.3)
# plt.savefig("acc曲线.svg", format='svg')
# plt.show()

