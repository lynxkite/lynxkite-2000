"""Operations for machine learning."""

import enum
import functools
import numpy as np
from . import core
from lynxkite.core import workspace
from .pytorch import pytorch_core
from lynxkite.core import ops
from tqdm import tqdm
import pandas as pd
import pathlib


op = ops.op_registration(core.ENV)


def load_ws(model_workspace: str):
    cwd = pathlib.Path()
    path = cwd / model_workspace
    assert path.is_relative_to(cwd), f"Path '{path}' is invalid"
    assert path.exists(), f"Workspace {path} does not exist"
    ws = workspace.Workspace.load(path)
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
    ws = load_ws(model_workspace + ".lynxkite.json")
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


def _get_num_samples(bundle: core.Bundle, input_mapping: pytorch_core.ModelMapping):
    """Returns the number of samples in the input mapping."""
    num_samples = None
    for k, v in input_mapping.map.items():
        if v.df in bundle.dfs and v.column in bundle.dfs[v.df]:
            if num_samples is None:
                num_samples = len(bundle.dfs[v.df][v.column])
            else:
                assert num_samples == len(bundle.dfs[v.df][v.column]), (
                    f"Input '{k}' has different number of samples ({len(bundle.dfs[v.df][v.column])}) "
                    f"than other inputs ({num_samples})."
                )
    return num_samples


@op("Train model", slow=True)
def train_model(
    bundle: core.Bundle,
    *,
    model_name: core.ModelName = "model",
    input_mapping: ModelTrainingInputMapping,
    epochs: int = 1,
    batch_size: int = 1,
):
    """
    Trains the selected model on the selected dataset.
    Training parameters specific to the model are set in the model definition,
    while parameters specific to the hardware environment and dataset are set here.
    """
    if input_mapping is None:
        return ops.Result(bundle, error="No inputs are selected.")
    m: pytorch_core.ModelConfig = bundle.other[model_name].copy()
    num_samples = _get_num_samples(bundle, input_mapping)
    if num_samples is None:
        return ops.Result(bundle, error="No inputs are selected.")
    num_batches = num_samples // batch_size
    tepochs = tqdm(range(epochs), desc="Training model")
    losses = []
    for _ in tepochs:
        total_loss = 0
        for i in tqdm(range(num_batches)):
            inputs = m.batch_tensors_from_bundle(
                bundle, batch_size, i, input_mapping
            )
            loss = m.train(inputs)
            total_loss += loss
        mean_loss = total_loss / len(inputs)
        tepochs.set_postfix({"loss": mean_loss})
        losses.append(mean_loss)
    m.trained = True
    bundle = bundle.copy()
    bundle.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle.other[model_name] = m
    return bundle


@op("Model inference", slow=True)
def model_inference(
    bundle: core.Bundle,
    *,
    model_name: core.ModelName = "model",
    input_mapping: ModelInferenceInputMapping,
    output_mapping: ModelOutputMapping,
    batch_size: int = 1,
):
    """Executes a trained model."""
    if input_mapping is None or output_mapping is None:
        return ops.Result(bundle, error="Mapping is unset.")
    m: pytorch_core.ModelConfig = bundle.other[model_name]
    assert m.trained, "The model is not trained."
    num_samples = _get_num_samples(bundle, input_mapping)
    if num_samples is None:
        return ops.Result(bundle, error="No inputs are selected.")
    num_batches = num_samples // batch_size
    outputs = {}
    for i in tqdm(range(num_batches)):
        inputs = m.batch_tensors_from_bundle(bundle, batch_size, i, input_mapping)
        batch_outputs = m.inference(inputs)
        for k, v in batch_outputs.items():
            v = v.detach().numpy().reshape(batch_size, -1)
            outputs.setdefault(k, []).extend(v.tolist())
    bundle = bundle.copy()
    copied = set()
    for k, v in output_mapping.map.items():
        if not v.df or not v.column:
            continue
        if v.df not in copied:
            bundle.dfs[v.df] = bundle.dfs[v.df].copy()
            copied.add(v.df)
        bundle.dfs[v.df][v.column] = outputs[k]
    return bundle


@op("Train/test split")
def train_test_split(
    bundle: core.Bundle, *, table_name: core.TableName, test_ratio: float = 0.1, seed=1234
):
    """Splits a dataframe in the bundle into separate "_train" and "_test" dataframes."""
    df = bundle.dfs[table_name]
    test = df.sample(frac=test_ratio, random_state=seed).reset_index()
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


class UMAPMetric(str, enum.Enum):
    l1 = "l1"
    cityblock = "cityblock"
    taxicab = "taxicab"
    manhattan = "manhattan"
    euclidean = "euclidean"
    l2 = "l2"
    sqeuclidean = "sqeuclidean"
    canberra = "canberra"
    minkowski = "minkowski"
    chebyshev = "chebyshev"
    linf = "linf"
    cosine = "cosine"
    correlation = "correlation"
    hellinger = "hellinger"
    hamming = "hamming"


@op("View vectors", view="visualization")
def view_vectors(
    bundle: core.Bundle,
    *,
    table_name: core.TableName = "nodes",
    vector_column: core.ColumnNameByTableName = "",
    label_column: core.ColumnNameByTableName = "",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: UMAPMetric = UMAPMetric.euclidean,
):
    try:
        from cuml.manifold.umap import UMAP  # ty: ignore[unresolved-import]
    except ImportError:
        from umap import UMAP
    vec = np.stack(bundle.dfs[table_name][vector_column].to_numpy())
    umap = functools.partial(
        UMAP,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric.value,
    )
    proj = umap(n_components=2).fit_transform(vec)
    color = umap(n_components=1).fit_transform(vec)
    data = [[*p.tolist(), "", c.item()] for p, c in zip(proj, color)]
    if label_column:
        for i, row in enumerate(bundle.dfs[table_name][label_column]):
            data[i][2] = row
    size = 100 / len(data) ** 0.4
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
