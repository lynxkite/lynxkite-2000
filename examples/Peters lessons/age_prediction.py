"""Custom operations for the Age prediction demo workspace."""

import pandas as pd
from tqdm import tqdm

from lynxkite_core import ops
from lynxkite_graph_analytics.operations.ml_ops import (
    ModelTrainingInputMapping,
    ModelOutputMapping,
    ModelInferenceInputMapping,
)
from lynxkite_graph_analytics.pytorch import pytorch_core
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "Age prediction")


@op("Train graph model", slow=True, icon="robot")
def train_graph_model(
    bundle: core.Bundle,
    *,
    model_name: pytorch_core.PyTorchModelName = "model",
    input_mapping: ModelTrainingInputMapping | None,
    epochs: int = 1,
):
    """
    Trains the selected model on the selected dataset without batching.
    :param bundle: the bundle
    :param model_name: the name of the model
    :param input_mapping: the input mapping
    :param epochs: the number of epochs
    """
    if input_mapping is None:
        return ops.Result(bundle, error="No inputs are selected.")
    m: pytorch_core.ModelConfig = bundle.other[model_name].copy()
    tepochs = tqdm(range(epochs), desc="Training graph model")
    losses = []
    input_ctx = pytorch_core.InputContext(batch_size=None, batch_index=0)
    for _ in tepochs:
        inputs = m.inputs_from_bundle(
            bundle,
            list(set(m.model_inputs) | set(m.loss_inputs) - set(m.model_outputs)),
            input_mapping,
            input_ctx,
        )
        loss = m.train(inputs)
        losses.append(loss)
        tepochs.set_postfix({"loss": loss})
    m.trained = True
    bundle = bundle.copy()
    bundle.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle.other[model_name] = m
    return bundle


def _tensor_to_column_values(tensor):
    values = tensor.detach().cpu().numpy()
    if values.ndim == 0:
        return values.item()
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values.tolist()


@op("Graph model inference", slow=True, icon="robot")
def graph_model_inference(
    bundle: core.Bundle,
    *,
    model_name: pytorch_core.PyTorchModelName = "model",
    input_mapping: ModelInferenceInputMapping | None,
    output_mapping: ModelOutputMapping | None,
    full_id_column: core.TableColumn,
    output_id_column: core.TableColumn,
):
    """
    Performs inference using the selected model.
    :param bundle: the bundle
    :param model_name: the name of the model
    :param input_mapping: the input mapping
    :param output_mapping: the output mapping
    :param full_id_column: the dataframe and id column for all the nodes in the graph
    :param output_id_column: the dataframe and id column for the nodes in the graph that are being predicted
    """
    if input_mapping is None or output_mapping is None:
        return ops.Result(bundle, error="Mapping is unset.")
    m: pytorch_core.ModelConfig = bundle.other[model_name]
    input_ctx = pytorch_core.InputContext(batch_size=None, batch_index=0)
    inputs = m.inputs_from_bundle(bundle, m.model_inputs, input_mapping, input_ctx)
    outputs = m.inference(inputs)
    bundle = bundle.copy()
    copied = set()
    for k, v in output_mapping.map.items():
        df = v.get("table_name")
        col = v.get("column")
        if not df or not col:
            continue
        if df not in copied:
            bundle.dfs[df] = bundle.dfs[df].copy()
            copied.add(df)
        values = _tensor_to_column_values(outputs[k])
        full_df_name = full_id_column[0]
        full_df = bundle.dfs[full_df_name]
        target_df = bundle.dfs[df]
        predictions_by_id = pd.Series(list(values), index=full_df[full_id_column[1]])
        bundle.dfs[df][col] = target_df[output_id_column[1]].map(predictions_by_id)
    return bundle


@op("Use numeric IDs for nodes in a relation", color="orange", icon="table-filled")
def numeric_id(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
):
    """Replaces the ids used in the relation with numeric ids. Useful when you want to put the ids into tensors.
    :param b: The bundle.
    :param relation_name: The name of the relation.
    """
    b = b.copy()
    b.dfs = b.dfs.copy()

    rel = next((r for r in b.relations if r.name == relation_name))

    source_df = b.dfs[rel.source_table].copy()
    target_df = b.dfs[rel.target_table].copy()
    edge_df = b.dfs[rel.name].copy()

    global_source_map = {old_id: idx for idx, old_id in enumerate(source_df[rel.source_key])}
    source_df[rel.source_key] = range(len(source_df))

    if rel.source_table == rel.target_table:
        global_target_map = global_source_map
        target_df[rel.target_key] = source_df[rel.source_key]
    else:
        global_target_map = {old_id: idx for idx, old_id in enumerate(target_df[rel.target_key])}
        target_df[rel.target_key] = range(len(target_df))

    edge_df[rel.source_column] = edge_df[rel.source_column].map(global_source_map)
    edge_df[rel.target_column] = edge_df[rel.target_column].map(global_target_map)

    b.dfs[rel.source_table] = source_df
    b.dfs[rel.target_table] = target_df
    b.dfs[rel.name] = edge_df

    for df_name, df in b.dfs.items():
        if df_name in (rel.source_table, rel.target_table, rel.name):
            continue

        if rel.source_key in df.columns:
            updated_df = df.copy()
            updated_df[rel.source_key] = updated_df[rel.source_key].map(global_source_map)
            b.dfs[df_name] = updated_df
    return b
