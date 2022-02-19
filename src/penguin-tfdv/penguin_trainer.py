from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_LABEL_KEY = 'species'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def _input_fn(
        file_pattern: List[str],
        data_accessor: tfx.components.DataAccessor,
        schema: schema_pb2.Schema,
        batch_size: int = 200) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema
    ).repeat()


def _build_keras_model(schema: schema_pb2.Schema) -> tf.keras.Model:
    # NEW: 此处由 schema 获取列名
    feature_keys = [f.name for f in schema.feature if f.name != _LABEL_KEY]
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in feature_keys]

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


def run_fn(fn_args: tfx.components.FnArgs):
    # NEW: 对比之前的 schema_utils.schema_from_feature_spec(_FEATURE_SPEC), 直接使用 SchemaGen 得出的结果
    schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, batch_size=_EVAL_BATCH_SIZE)

    # NEW: _build_keras_model 加了 schema 的输入参数
    model = _build_keras_model(schema)
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )
    model.save(fn_args.serving_model_dir, save_format='tf')
