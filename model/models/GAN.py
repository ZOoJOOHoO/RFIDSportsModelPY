import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input
from tensorflow.keras.models import Model
import numpy as np
import model.C_dataForTrainMaker as loadData
import os

def build_generator(latent_dim, time_steps, num_features):
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(time_steps * num_features, activation='sigmoid'))  # 使用 sigmoid 激活函数确保输出在 0 和 1 之间
    model.add(Reshape((time_steps, num_features)))  # reshape to (time_steps, num_features)
    return model

def build_discriminator(time_steps, num_features):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(time_steps, num_features)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def train_gan(generator, discriminator, gan, data, epochs, batch_size, latent_dim):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # 训练判别器
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

def generate_data(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)
    return generated_data

time_steps = 90
num_features = 8
latent_dim = 100
epochs = 10000
batch_size = 64


# 构建和编译模型
generator = build_generator(latent_dim, time_steps, num_features)
discriminator = build_discriminator(time_steps, num_features)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

x,y=loadData.getDataSpecialPosition("D:\py_project\RfidSport\model\dataSets", isInvert=1,position=1)
x = x[:, :8, :]

# 训练GAN
train_gan(generator, discriminator, gan, x, epochs, batch_size, latent_dim)

# 生成新的数据
new_data = generate_data(generator, latent_dim, 100)
print(new_data.shape)  # 输出 (100, 90, 8)


