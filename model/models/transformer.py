import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.embed_dim = embed_dim

    def get_angles(self, position, i, embed_dim):
        angles = 1 / tf.pow(10000.0, (2 * tf.cast(i, tf.float32) / tf.cast(embed_dim, tf.float32)))
        return tf.cast(position, tf.float32) * angles

    def positional_encoding(self, max_position, embed_dim):
        angle_rads = self.get_angles(
            tf.range(max_position)[:, tf.newaxis],
            tf.range(embed_dim)[tf.newaxis, :],
            embed_dim
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(seq_len, self.embed_dim)
        return inputs + pos_encoding


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % self.num_heads == 0
        self.projection_dim = embed_dim // self.num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, input_vocab_size, output_vocab_size, max_seq_len,
                 rate=0.1):
        super(Transformer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(output_vocab_size, activation='softmax')

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)
        x = self.flatten(x)
        return self.dense(x)


# Hyperparameters
num_layers = 2
embed_dim = 32
num_heads = 4
ff_dim = 64
input_vocab_size = 100000
output_vocab_size = 2
max_seq_len = 90
rate = 0.1

scaler = MinMaxScaler()
category_1_data = np.random.randint(1, 1000, size=(100, 16, max_seq_len))
category_1_data = category_1_data.reshape(500, -1)
category_1_labels = np.full((500,), 0)

category_2_data = np.random.randint(1, 1100, size=(100, 16, max_seq_len))
category_2_data = category_2_data.reshape(500, -1)
category_2_labels = np.full((500,), 1)


# 将两个类别的数据和标签合并
X_train = np.vstack((category_1_data, category_2_data))
y_train = np.hstack((category_1_labels, category_2_labels))

# 打乱数据
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and compile the model
model = Transformer(num_layers, embed_dim, num_heads, ff_dim, input_vocab_size, output_vocab_size, max_seq_len*16, rate)
# 设置学习率为0.01
custom_adam_optimizer = Adam(learning_rate=0.001)

# 使用自定义优化器编译模型
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

model.compile(optimizer=custom_adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model