import math
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.datasets as datasets

input_layer = keras.Input(shape=(4,))
hide1_layer = keras.layers.Dense(units=8, activation='relu')
hide2_layer = keras.layers.Dense(units=4, activation='relu')
output_layer = keras.layers.Dense(units=1, activation='sigmoid')

model = keras.Sequential(layers=[input_layer, hide1_layer, hide2_layer, output_layer])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

data, target = datasets.load_iris(return_X_y=True)
data = data[:100, :]   # 取前面100个样本（第一类与第二类）
target = target[:100]

epochs = 20
batch_size = 10
batch_num = int(math.ceil(len(data) / batch_size))

optimizer = keras.optimizers.Adam()
losser = keras.losses.BinaryCrossentropy()

for epoch in range(epochs):
    print("loop-{}".format(epoch))
    for i in range(batch_num):
        with tf.GradientTape() as tape:
            start_ = i * batch_size
            end_ = (i + 1) * batch_size
            predictions = model(data[start_: end_])
            loss_value = losser(target[start_: end_], predictions[:, 0])
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

pre_result = model(data)
category = [0 if item <= 0.5 else 1 for item in pre_result]
accuracy = (target == category).mean()
print(F'分类准确度：{accuracy *100.0:5.2f}%')
