"""
https://tensorflow.google.cn/tfx/tutorials/tfx/penguin_tfdv
"""
import commandr

import os
from tfx import v1 as tfx
from absl import logging
logging.set_verbosity(logging.INFO)

PROJECT_ROOT = os.path.join(os.environ['HOME'], 'private_workspace/demo-tfx')

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/penguin-simple')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')

SCHEMA_PIPELINE_NAME = "penguin-tfdv-schema"
PIPELINE_NAME = "penguin-tfdv"

SCHEMA_PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, SCHEMA_PIPELINE_NAME, 'pipelines')
SCHEMA_METADATA_PATH = os.path.join(OUTPUT_ROOT, SCHEMA_PIPELINE_NAME, 'metadata', 'metadata.db')
PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'pipelines')
METADATA_PATH = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'metadata', 'metadata.db')
SERVING_MODEL_DIR = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'serving_model')

_trainer_module_file = os.path.join(PROJECT_ROOT, 'src/penguin-tfdv', 'penguin_trainer.py')


@commandr.command
def prepare():
    def _create_schema_pipeline(
            pipeline_name: str,
            pipeline_root: str,
            data_root: str,
            metadata_path: str
            ) -> tfx.dsl.Pipeline:
        example_gen = tfx.components.CsvExampleGen(input_base=data_root)

        # NEW: Computes statistics over data for visualization and schema generation.
        statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
        # NEW: Generates schema based on the generated statistics.
        schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

        components = [example_gen, statistics_gen, schema_gen, ]

        return tfx.dsl.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
            components=components
        )

    pipeline = _create_schema_pipeline(
        pipeline_name=SCHEMA_PIPELINE_NAME,
        pipeline_root=SCHEMA_PIPELINE_ROOT,
        data_root=DATA_ROOT,
        metadata_path=SCHEMA_METADATA_PATH
    )
    tfx.orchestration.LocalDagRunner().run(pipeline)


@commandr.command
def train():
    def get_latest_schema_path():
        path_prefix = os.path.join(SCHEMA_PIPELINE_ROOT, 'SchemaGen/schema')
        for _, folders, _ in os.walk(path_prefix):
            max_id = max(map(int, folders))
            return os.path.join(path_prefix, str(max_id))

    def _create_pipeline(
            pipeline_name: str, pipeline_root: str, data_root: str,
            schema_path: str, module_file: str, serving_model_dir: str,
            metadata_path: str
    ) -> tfx.dsl.Pipeline:
        # 输入数据
        example_gen = tfx.components.CsvExampleGen(input_base=data_root)

        # NEW: 对新数据做统计
        statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
        # NEW: 读取之前生成的 schema
        schema_importer = tfx.dsl.Importer(
            source_uri=schema_path, artifact_type=tfx.types.standard_artifacts.Schema
        ).with_id('schema_importer')
        # NEW: 基于之前的 schema 和当前数据的统计结果做异常检测
        example_validator = tfx.components.ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
            schema=schema_importer.outputs['result']
        )

        trainer = tfx.components.Trainer(
            module_file=module_file,
            examples=example_gen.outputs['examples'],
            schema=schema_importer.outputs['result'],  # NEW: 使用已有的 schema
            train_args=tfx.proto.TrainArgs(num_steps=100),
            eval_args=tfx.proto.EvalArgs(num_steps=5)
        )

        pusher = tfx.components.Pusher(
            model=trainer.outputs['model'],
            push_destination=tfx.proto.PushDestination(
                filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)
            )
        )

        components = [
            example_gen,
            statistics_gen, schema_importer, example_validator,  # NEW: 新增
            trainer, pusher,
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
        schema_path=get_latest_schema_path(),
        module_file=_trainer_module_file,
        serving_model_dir=SERVING_MODEL_DIR,
        metadata_path=METADATA_PATH
    )
    tfx.orchestration.LocalDagRunner().run(pipeline)


@commandr.command('inspect')
def inspect_saved_model():
    import tensorflow as tf

    # Find a model with the latest timestamp.
    model_dirs = (item for item in os.scandir(SERVING_MODEL_DIR) if item.is_dir())
    model_path = max(model_dirs, key=lambda i: int(i.name)).path

    print(model_path)
    loaded_model = tf.keras.models.load_model(model_path)
    inference_fn = loaded_model.signatures['serving_default']
    print(inference_fn)


if __name__ == '__main__':
    commandr.Run()
