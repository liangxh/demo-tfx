import os
from tfx import v1 as tfx
from tfx.types import standard_component_specs
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.metadata import Metadata
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.orchestration.experimental.interactive import visualizations
from tfx.orchestration.experimental.interactive import standard_visualizations

SCHEMA_PIPELINE_NAME = "penguin-tfdv-schema"
PIPELINE_NAME = "penguin-tfdv"

PROJECT_ROOT = os.path.join(os.environ['HOME'], 'private_workspace/demo-tfx')

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/penguin-simple')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output')

SCHEMA_PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, SCHEMA_PIPELINE_NAME, 'pipelines')
SCHEMA_METADATA_PATH = os.path.join(OUTPUT_ROOT, SCHEMA_PIPELINE_NAME, 'metadata', 'metadata.db')
PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'pipelines')
METADATA_PATH = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'metadata', 'metadata.db')
SERVING_MODEL_DIR = os.path.join(OUTPUT_ROOT, PIPELINE_NAME, 'serving_model')


def get_latest_artifacts(metadata, pipeline_name, component_id):
    """Output artifacts of the latest run of the component."""
    context = metadata.store.get_context_by_type_and_name('node', f'{pipeline_name}.{component_id}')
    executions = metadata.store.get_executions_by_context(context.id)
    latest_execution = max(executions, key=lambda e: e.last_update_time_since_epoch)
    return execution_lib.get_artifacts_dict(metadata, latest_execution.id, [metadata_store_pb2.Event.OUTPUT])


def visualize_artifacts(artifacts):
    """Visualizes artifacts using standard visualization modules."""
    for artifact in artifacts:
        visualization = visualizations.get_registry().get_visualization(artifact.type_name)
        if visualization:
            visualization.display(artifact)


standard_visualizations.register_standard_visualizations()

# visualize

metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(SCHEMA_METADATA_PATH)

with Metadata(metadata_connection_config) as metadata_handler:
    stat_gen_output = get_latest_artifacts(metadata_handler, SCHEMA_PIPELINE_NAME, 'StatisticsGen')
    stats_artifacts = stat_gen_output[standard_component_specs.STATISTICS_KEY]

    schema_gen_output = get_latest_artifacts(metadata_handler, SCHEMA_PIPELINE_NAME, 'SchemaGen')
    schema_artifacts = schema_gen_output[standard_component_specs.SCHEMA_KEY]

visualize_artifacts(stats_artifacts)

visualize_artifacts(schema_artifacts)


