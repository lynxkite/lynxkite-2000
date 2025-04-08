"""Operations for machine learning."""

import numpy as np
from . import core
from lynxkite.core import workspace
from .pytorch import pytorch_core
from lynxkite.core import ops
from tqdm import tqdm
import cuml
import joblib
import pandas as pd
import pathlib


mem = joblib.Memory(".joblib-cache")
op = ops.op_registration(core.ENV)


def load_ws(model_workspace: str):
    cwd = pathlib.Path()
    path = cwd / model_workspace
    assert path.is_relative_to(cwd)
    assert path.exists(), f"Workspace {path} does not exist"
    ws = workspace.load(path)
    return ws


@op("Define model")
def define_model(
    bundle: core.Bundle,
    *,
    model_workspace: str,
    save_as: str = "model",
):
    """Trains the selected model on the selected dataset. Most training parameters are set in the model definition."""
    assert model_workspace, "Model workspace is unset."
    ws = load_ws(model_workspace)
    # Build the model without inputs, to get its interface.
    m = pytorch_core.build_model(ws)
    m.source_workspace = model_workspace
    bundle = bundle.copy()
    bundle.other[save_as] = m
    return bundle


# These contain the same mapping, but they get different UIs.
# For inputs, you select existing columns. For outputs, you can create new columns.
class ModelInferenceInputMapping(pytorch_core.ModelMapping):
    pass


class ModelTrainingInputMapping(pytorch_core.ModelMapping):
    pass


class ModelOutputMapping(pytorch_core.ModelMapping):
    pass


@op("Train model")
@ops.slow
def train_model(
    bundle: core.Bundle,
    *,
    model_name: str = "model",
    input_mapping: ModelTrainingInputMapping,
    epochs: int = 1,
):
    """Trains the selected model on the selected dataset. Most training parameters are set in the model definition."""
    m = bundle.other[model_name].copy()
    inputs = pytorch_core.to_tensors(bundle, input_mapping)
    t = tqdm(range(epochs), desc="Training model")
    losses = []
    for _ in t:
        loss = m.train(inputs)
        t.set_postfix({"loss": loss})
        losses.append(loss)
    m.trained = True
    bundle = bundle.copy()
    bundle.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle.other[model_name] = m
    return bundle


@op("Model inference")
@ops.slow
def model_inference(
    bundle: core.Bundle,
    *,
    model_name: str = "model",
    input_mapping: ModelInferenceInputMapping,
    output_mapping: ModelOutputMapping,
):
    """Executes a trained model."""
    if input_mapping is None or output_mapping is None:
        return ops.Result(bundle, error="Mapping is unset.")
    m = bundle.other[model_name]
    assert m.trained, "The model is not trained."
    inputs = pytorch_core.to_tensors(bundle, input_mapping)
    outputs = m.inference(inputs)
    bundle = bundle.copy()
    copied = set()
    for k, v in output_mapping.map.items():
        if not v.df or not v.column:
            continue
        if v.df not in copied:
            bundle.dfs[v.df] = bundle.dfs[v.df].copy()
            copied.add(v.df)
        bundle.dfs[v.df][v.column] = outputs[k].detach().numpy().tolist()
    return bundle


@op("Train/test split")
def train_test_split(bundle: core.Bundle, *, table_name: str, test_ratio: float = 0.1):
    """Splits a dataframe in the bundle into separate "_train" and "_test" dataframes."""
    df = bundle.dfs[table_name]
    test = df.sample(frac=test_ratio).reset_index()
    train = df.drop(test.index).reset_index()
    bundle = bundle.copy()
    bundle.dfs[f"{table_name}_train"] = train
    bundle.dfs[f"{table_name}_test"] = test
    return bundle


@op("View loss", view="visualization")
def view_loss(bundle: core.Bundle):
    loss = bundle.dfs["training"].training_loss.tolist()
    v = {
        "title": {"text": "Training loss"},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"data": loss, "type": "line"}],
    }
    return v


@op("Reduce dimensions")
def reduce_dimensions(bundle: core.Bundle):
    return


VIRIDIS = [
    "#440154",
    "#482777",
    "#3E4989",
    "#31688E",
    "#26828E",
    "#1F9E89",
    "#35B779",
    "#6CCE59",
    "#B4DE2C",
    "#FDE725",
]


@op("View vectors", view="visualization")
def view_vectors(
    bundle: core.Bundle,
    *,
    table_name: str = "nodes",
    vector_column: str = "",
    label_column: str = "",
):
    vec = np.stack(bundle.dfs[table_name][vector_column].to_numpy())
    proj = cuml.manifold.umap.UMAP(n_components=2).fit_transform(vec)
    color = cuml.manifold.umap.UMAP(n_components=1).fit_transform(vec)
    data = [[*p.tolist(), "", c.item()] for p, c in zip(proj, color)]
    if label_column:
        for i, row in enumerate(bundle.dfs[table_name][label_column]):
            data[i][2] = row
    size = 50 / len(data) ** 0.5
    v = {
        "title": {
            "text": f"UMAP projection of {vector_column}",
        },
        "visualMap": {
            "min": color[:, 0].min().item(),
            "max": color[:, 0].max().item(),
            "right": 10,
            "top": "center",
            "calculable": True,
            "dimension": 3,
            "inRange": {"color": VIRIDIS},
        },
        "tooltip": {"trigger": "item", "formatter": "GET_THIRD_VALUE"}
        if label_column
        else {"show": False},
        "xAxis": [{"type": "value"}],
        "yAxis": [{"type": "value"}],
        "series": [{"type": "scatter", "symbolSize": size, "data": data}],
    }
    return v
