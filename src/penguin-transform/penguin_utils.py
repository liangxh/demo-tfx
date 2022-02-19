from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEYS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
_LABEL_KEY = 'species'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


# NEW: 由 Transform 调用
def preprocessing_fn(inputs):
    outputs = {}

    # 标准化
    for key in _FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # 完成 species 字段到 ID 的转换, 类别多的话可以用 tft.compute_and_apply_vocabulary()
    table_keys = ['Adelie', 'Chinstrap', 'Gentoo']
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys, values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype=tf.string, value_dtype=tf.int64
    )
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[_LABEL_KEY] = table.lookup(inputs[_LABEL_KEY])

    return outputs


# def _apply_preprocessing(raw_features, tft_layer):
#     transformed_features = tft_layer(raw_features)
#     transformed_label = transformed_features.pop(_LABEL_KEY, None)
#     return transformed_features, transformed_label


# NEW: 分別供 _input_fn 和 _get_serve_tf_examples_fn 对训练数据和预测数据做数据预处理
def _get_transform_fn(tft_layer):
    def transform_fn(raw_features):
        transformed_features = tft_layer(raw_features)
        transformed_label = transformed_features.pop(_LABEL_KEY, None)
        return transformed_features, transformed_label
    return transform_fn


# NEW: 返回一个 handler 函数，输入为序列化的 tf.example, 输出为预测结果
def _get_serve_tf_examples_fn(model, tf_transform_output):
    # 保存 transform 层
    model.tft_layer = tf_transform_output.transform_features_layer()
    _transform_fn = _get_transform_fn(model.tft_layer)

    # 只选取需要的特征
    feature_spec = tf_transform_output.raw_feature_spec()
    required_feature_spec = {k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_examples):
        # 解析序列化的 examples
        parsed_features = tf.io.parse_example(serialized_tf_examples, required_feature_spec)
        # 做 transform 转换
        transformed_features, _ = _transform_fn(parsed_features)
        # 做预测
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,  # 由原本的 schema 改成了 transform 的輸出
              batch_size: int = 200) -> tf.data.Dataset:
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema=tf_transform_output.raw_metadata.schema
    )

    transform_layer = tf_transform_output.transform_features_layer()
    _transform_fn = _get_transform_fn(transform_layer)

    return dataset.map(_transform_fn).repeat()


def _build_keras_model() -> tf.keras.Model:
    inputs = [keras.layers.Input(shape=(1,), name=key) for key in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for _ in range(2):
        d = keras.layers.Dense(8, activation='relu')(d)
    outputs = keras.layers.Dense(3)(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    model.summary(print_fn=logging.info)
    return model


# 由 Trainer 调用
def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output, batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output, batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    # NEW: 把 transform graph 也加入
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
