import commandr
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def load_dataset(filename='data/penguin-simple/data.csv'):
    data = list()
    target = list()
    with open(filename, 'r') as file_obj:
        for line in file_obj:
            if line.startswith('s'):
                continue
            line = line.strip()
            if line == '':
                continue
            parts = line.split(',')
            x = list(map(float, parts[1:]))
            y = int(parts[0])
            data.append(x)
            target.append(y)
    return np.asarray(data), np.asarray(target)


@commandr.command
def train():
    data, target = load_dataset()

    input_layer = keras.Input(shape=(4,))
    hide1_layer = keras.layers.Dense(units=8, activation='relu')
    hide2_layer = keras.layers.Dense(units=4, activation='relu')
    output_layer = keras.layers.Dense(units=3, activation='softmax')

    '''
    hide1_layer_tensor = hide1_layer(input_layer)
    hide2_layer_tensor = hide2_layer(hide1_layer_tensor)
    output_layer_tensor = output_layer(hide2_layer_tensor)
    model = keras.Model(inputs=input_layer, outputs=output_layer_tensor)
    '''

    model = keras.Sequential(layers=[input_layer, hide1_layer, hide2_layer, output_layer])

    metrics = [
        keras.metrics.SparseCategoricalAccuracy()
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics
    )

    model.fit(
        x=data,
        y=target,
        batch_size=10,
        epochs=10,
        verbose=True,
        shuffle=True
    )

    pre_result = model.predict(data)
    print(pre_result)
    predicted_labels = pre_result.argmax(axis=1)
    print(predicted_labels)

    accuracy = (target == predicted_labels).mean()
    print(F'分类准确度：{accuracy *100.0:5.2f}%', )

    # model.save('output/debug-penguing-simple', save_format='tf')


if __name__ == '__main__':
    commandr.Run()
