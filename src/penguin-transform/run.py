import commandr
import os
from absl import logging
logging.set_verbosity(logging.INFO)

PIPELINE_NAME = "penguin-transform"

PROJECT_ROOT = os.path.join(os.environ['HOME'], 'private_workspace/demo-tfx')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')

PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'pipelines')
METADATA_PATH = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'metadata', 'metadata.db')
SERVING_MODEL_DIR = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'serving_model')

# https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_size.csv
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/penguin-size/data')
# https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/schema/raw/schema.pbtxt
SCHEMA_PATH = os.path.join(PROJECT_ROOT, 'data/penguin-size/schema')

_module_file = os.path.join(PROJECT_ROOT, 'src/penguin-transform', 'penguin_utils.py')


@commandr.command
def main():
    from tfx import v1 as tfx

    def _create_pipeline(
            pipeline_name: str, pipeline_root: str, data_root: str,
            schema_path: str, module_file: str, serving_model_dir: str,
            metadata_path: str) -> tfx.dsl.Pipeline:
        example_gen = tfx.components.CsvExampleGen(input_base=data_root)
        # 同 penguin-tfdv
        statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
        # 同 penguin-tfdv
        schema_importer = tfx.dsl.Importer(
            source_uri=schema_path, artifact_type=tfx.types.standard_artifacts.Schema
        ).with_id('schema_importer')

        # Performs anomaly detection based on statistics and data schema.
        example_validator = tfx.components.ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
            schema=schema_importer.outputs['result']
        )

        # NEW: 原来直接传给 trainer 的 schema 输入
        transform = tfx.components.Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_importer.outputs['result'],
            materialize=False,
            module_file=module_file
        )

        trainer = tfx.components.Trainer(
            module_file=module_file,
            examples=example_gen.outputs['examples'],

            # NEW: tfdv 中用的是 schema=schema_importer.outputs['result']
            transform_graph=transform.outputs['transform_graph'],

            train_args=tfx.proto.TrainArgs(num_steps=100),
            eval_args=tfx.proto.EvalArgs(num_steps=5)
        )

        # Pushes the model to a filesystem destination.
        pusher = tfx.components.Pusher(
            model=trainer.outputs['model'],
            push_destination=tfx.proto.PushDestination(
                filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)
            )
        )

        components = [
            example_gen, statistics_gen, schema_importer,  example_validator,
            transform,  # NEW: Transform component was added to the pipeline.
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
        schema_path=SCHEMA_PATH,
        module_file=_module_file,
        serving_model_dir=SERVING_MODEL_DIR,
        metadata_path=METADATA_PATH
    )
    tfx.orchestration.LocalDagRunner().run(pipeline)


@commandr.command('inspect-cmd')
def show_saved_model():
    print(f'saved_model_cli show --dir {SERVING_MODEL_DIR}/$(ls -1 {SERVING_MODEL_DIR} | sort -nr | head -1) --tag_set serve --signature_def serving_default')


"""
The given SavedModel SignatureDef contains the following input(s):
  inputs['examples'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: serving_default_examples:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 3)
      name: StatefulPartitionedCall_2:0
Method name is: tensorflow/serving/predict
"""


@commandr.command('inspect')
def inspect_saved_model():
    import tensorflow as tf

    # SERVING_MODEL_DIR 底下没有 keras_metadata.pb
    path = os.path.join(PIPELINE_ROOT, 'Pusher/pushed_model')

    model_dirs = (item for item in os.scandir(path) if item.is_dir())
    model_path = max(model_dirs, key=lambda i: int(i.name)).path

    print(model_path)
    loaded_model = tf.keras.models.load_model(model_path)
    inference_fn = loaded_model.signatures['serving_default']

    def load_dataset():
        required_columns = ['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
        dataset = {col: list() for col in required_columns}

        with open(os.path.join(DATA_ROOT, 'data.csv'), 'r') as file_obj:
            first_line = file_obj.readline()
            columns = first_line.strip().split(',')
            col_idx_map = {column: i for i, column in enumerate(columns)}
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split(',')
                for col in required_columns:
                    idx = col_idx_map[col]
                    dataset[col].append(parts[idx])
        return dataset

    label_to_idx = lambda _label: {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}[_label]

    dataset = load_dataset()
    culmen_length_mm = list(map(float, dataset['culmen_length_mm']))
    culmen_depth_mm = list(map(float, dataset['culmen_depth_mm']))
    flipper_length_mm = list(map(int, dataset['flipper_length_mm']))
    body_mass_g = list(map(int, dataset['body_mass_g']))
    species = list(map(label_to_idx, dataset['species']))

    # culmen_length_mm = [49.9]
    # culmen_depth_mm = [16.1]
    # flipper_length_mm = [213]
    # body_mass_g = [5400]
    # species = [2]

    hit_count = 0.

    for _specie, _culmen_length_mm, _culmen_depth_mm, _flipper_length_mm, _body_mass_g \
            in zip(species, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g):
        features = {
          'culmen_length_mm': tf.train.Feature(float_list=tf.train.FloatList(value=[_culmen_length_mm, ])),
          'culmen_depth_mm': tf.train.Feature(float_list=tf.train.FloatList(value=[_culmen_depth_mm, ])),
          'flipper_length_mm': tf.train.Feature(int64_list=tf.train.Int64List(value=[_flipper_length_mm])),
          'body_mass_g': tf.train.Feature(int64_list=tf.train.Int64List(value=[_body_mass_g])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        examples = example_proto.SerializeToString()

        result = inference_fn(examples=tf.constant([examples]))
        hit_count += result['output_0'].numpy().argmax() == _specie

    print(f'accuracy: {hit_count / len(species)}')


if __name__ == '__main__':
    commandr.Run()
