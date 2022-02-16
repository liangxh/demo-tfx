"""
https://tensorflow.google.cn/tfx/tutorials/tfx/penguin_simple
"""
import os
import tensorflow as tf
from tfx import v1 as tfx
from absl import logging

logging.set_verbosity(logging.INFO)  # Set default logging level.

print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))

PIPELINE_NAME = "penguin-simple"

DATA_ROOT = 'data/{}'.format(PIPELINE_NAME)
# data/data.csv
# https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv

OUTPUT_ROOT = 'output'
PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, 'pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_ROOT, 'metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(OUTPUT_ROOT, 'serving_model', PIPELINE_NAME)

_trainer_module_file = os.path.join(
    os.environ['HOME'], 'private_workspace/demo-tfx/src/penguin-simple', 'penguin_trainer.py'
)


def _create_pipeline(
        pipeline_name: str, pipeline_root: str, data_root: str,
        module_file: str, serving_model_dir: str,
        metadata_path: str) -> tfx.dsl.Pipeline:
    # 指定输入数据
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)
    # 指定模型训练脚本和参数
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5)
    )
    # 推送输出文件到目标路径
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)
        )
    )
    # Following three components will be included in the pipeline.
    components = [
        example_gen,
        trainer,
        pusher,
    ]
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        components=components
    )


pipeline = _create_pipeline(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    data_root=DATA_ROOT,
    module_file=_trainer_module_file,
    serving_model_dir=SERVING_MODEL_DIR,
    metadata_path=METADATA_PATH
)
tfx.orchestration.LocalDagRunner().run(pipeline)
