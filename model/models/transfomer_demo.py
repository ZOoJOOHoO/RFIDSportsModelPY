# @Time    : 2024/5/13 21:15
# @Author  : ZJH
# @FileName: transfomer_demo.py
# @Software: PyCharm
# 导入 TensorFlow 和 Keras 相关的库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import model.C_dataForTrainMaker as loadData
from tensorflow.keras.regularizers import l2

class TransformerBlock(layers.Layer):
    # 类的初始化方法
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个多头注意力（MultiHeadAttention）层
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # 创建一个前馈网络（Feed Forward Network），包含两个全连接层
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu",kernel_regularizer=l2(0.01)), layers.Dense(embed_dim,kernel_regularizer=l2(0.01)),]
        )
        # 创建两个层归一化（LayerNormalization）层
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # 创建两个 dropout 层，用于防止过拟合
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # 定义 call 方法，用于处理输入
    def call(self, inputs, training):
        # 应用多头注意力机制
        attn_output = self.att(inputs, inputs)
        # 应用第一个 dropout 层
        attn_output = self.dropout1(attn_output, training=training)
        # 应用第一个层归一化，并进行残差连接
        out1 = self.layernorm1(inputs + attn_output)
        # 通过前馈网络
        ffn_output = self.ffn(out1)
        # 应用第二个 dropout 层
        ffn_output = self.dropout2(ffn_output, training=training)
        # 应用第二个层归一化，并进行残差连接
        return self.layernorm2(out1 + ffn_output)

# 定义一个 TokenAndPositionEmbedding 类，继承自 Keras 的 Layer 类
class TokenAndPositionEmbedding(layers.Layer):
    # 类的初始化方法
    def __init__(self, maxlen, vocab_size, embed_dim):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个词嵌入层，用于将词汇标识符转换为嵌入向量
        # vocab_size 是词汇表的大小，embed_dim 是嵌入向量的维度
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # 创建一个位置嵌入层，用于将位置信息转换为嵌入向量
        # maxlen 是序列的最大长度，embed_dim 是嵌入向量的维度
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    # 定义 call 方法，用于处理输入
    def call(self, x):
        # 获取输入序列 x 的长度
        maxlen = tf.shape(x)[-1]
        # 生成一个从 0 到 maxlen 的位置索引序列
        positions = tf.range(start=0, limit=maxlen, delta=1)
        # 通过位置嵌入层将位置索引转换为嵌入向量
        positions = self.pos_emb(positions)
        # 通过词嵌入层将输入序列 x 转换为嵌入向量
        x = self.token_emb(x)
        return x + positions

vocab_size = 1001
maxlen = 16*90

X_train, X_test, Y_train, Y_test = loadData.getXandY("D:\py_project\RfidSport\model\dataSets", isInvert=0)
X_train = np.floor(X_train[:, 0:16, :] * 1000)
X_test = np.floor(X_test[:, 0:16, :] * 1000)
# shape = X_train[:, 0:8, :].shape
# pi_array = np.full(shape, np.pi)
# X_train = (X_train[:, 0:8, :] + pi_array) * 1000
# shape = X_test[:, 0:8, :].shape
# pi_array = np.full(shape, np.pi)
# X_test = (X_test[:, 0:8, :] + pi_array) * 1000
X_train = X_train.astype(np.int32)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.astype(np.int32)
X_test = X_test.reshape(X_test.shape[0], -1)

# 设置每个 token 的嵌入维度
embed_dim = 16
# 设置注意力机制中的头数
num_heads = 2
# 设置 Transformer 块内前馈网络的隐藏层大小
ff_dim = 16

# 定义模型的输入，指定输入序列的最大长度
inputs = layers.Input(shape=(maxlen,))
# 创建一个 Token 和位置嵌入层，用于处理输入序列
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
# 将输入数据传递给嵌入层
x = embedding_layer(inputs)
# 创建一个 Transformer 块
transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim)
# 将嵌入后的输入数据传递给 Transformer 块
x = transformer_block1(x)
x = transformer_block2(x)
#x = transformer_block3(x)
# 使用全局平均池化层，以减少维度并处理变长输入
x = layers.GlobalAveragePooling1D()(x)
# 应用 Dropout 层，用于减少过拟合
x = layers.Dropout(0.1)(x)
# 添加一个全连接层，激活函数为 relu
x = layers.Dense(20, activation="relu", kernel_regularizer=l2(0.01))(x)
# 再次应用 Dropout 层
x = layers.Dropout(0.1)(x)
# 最后一个全连接层，输出维度为 2，激活函数为 softmax，用于分类
outputs = layers.Dense(5, activation="softmax")(x)

tf.random.set_seed(65)
# 创建模型，指定输入和输出
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    X_train, Y_train, batch_size=8, epochs=100, validation_data=(X_test, Y_test)
)