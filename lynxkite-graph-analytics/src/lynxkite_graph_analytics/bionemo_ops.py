"""BioNeMo related operations

The intention is to showcase how BioNeMo can be integrated with LynxKite. This should be
considered as a reference implementation and not a production ready code.
The operations are quite specific for this example notebook:
https://github.com/NVIDIA/bionemo-framework/blob/main/docs/docs/user-guide/examples/bionemo-geneformer/geneformer-celltype-classification.ipynb
"""

from lynxkite.core import ops
import requests
import tarfile
import os
from collections import Counter
from . import core
import joblib
import numpy as np
import torch
from pathlib import Path
import random
from contextlib import contextmanager
import cellxgene_census  # TODO: This needs numpy < 2
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from bionemo.scdl.io.single_cell_collection import SingleCellCollection

import scanpy


mem = joblib.Memory(".joblib-cache")
op = ops.op_registration(core.ENV)
DATA_PATH = Path("/workspace")


@contextmanager
def random_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        # Go back to previous state
        random.setstate(state)


@op("BioNeMo > Download CELLxGENE dataset")
@mem.cache()
def download_cellxgene_dataset(
    *,
    save_path: str,
    census_version: str = "2023-12-15",
    organism: str = "Homo sapiens",
    value_filter='dataset_id=="8e47ed12-c658-4252-b126-381df8d52a3d"',
    max_workers: int = 1,
    use_mp: bool = False,
) -> None:
    """Downloads a CELLxGENE dataset"""

    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism,
            obs_value_filter=value_filter,
        )
    with random_seed(32):
        indices = list(range(len(adata)))
        random.shuffle(indices)
    micro_batch_size: int = 32
    num_steps: int = 256
    selection = sorted(indices[: micro_batch_size * num_steps])
    # NOTE: there's a current constraint that predict_step needs to be a function of micro-batch-size.
    #  this is something we are working on fixing. A quick hack is to set micro-batch-size=1, but this is
    #  slow. In this notebook we are going to use mbs=32 and subsample the anndata.
    adata = adata[selection].copy()  # so it's not a view
    h5ad_outfile = DATA_PATH / Path("hs-celltype-bench.h5ad")
    adata.write_h5ad(h5ad_outfile)
    with tempfile.TemporaryDirectory() as temp_dir:
        coll = SingleCellCollection(temp_dir)
        coll.load_h5ad_multi(
            h5ad_outfile.parent, max_workers=max_workers, use_processes=use_mp
        )
        coll.flatten(DATA_PATH / save_path, destroy_on_copy=True)
    return DATA_PATH / save_path


@op("BioNeMo > Import H5AD file")
def import_h5ad(*, file_path: str):
    return scanpy.read_h5ad(DATA_PATH / Path(file_path))


@op("BioNeMo > Download model")
@mem.cache(verbose=1)
def download_model(*, model_name: str) -> str:
    """Downloads a model."""
    model_download_parameters = {
        "geneformer_100m": {
            "name": "geneformer_100m",
            "version": "2.0",
            "path": "geneformer_106M_240530_nemo2",
        },
        "geneformer_10m": {
            "name": "geneformer_10m",
            "version": "2.0",
            "path": "geneformer_10M_240530_nemo2",
        },
        "geneformer_10m2": {
            "name": "geneformer_10m",
            "version": "2.1",
            "path": "geneformer_10M_241113_nemo2",
        },
    }

    # Define the URL and output file
    url_template = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/{name}/{version}/files?redirect=true&path={path}.tar.gz"
    url = url_template.format(**model_download_parameters[model_name])
    model_filename = f"{DATA_PATH}/{model_download_parameters[model_name]['path']}"
    output_file = f"{model_filename}.tar.gz"

    # Send the request
    response = requests.get(url, allow_redirects=True, stream=True)
    response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)

    # Save the file to disk
    with open(f"{output_file}", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Extract the tar.gz file
    os.makedirs(model_filename, exist_ok=True)
    with tarfile.open(output_file, "r:gz") as tar:
        tar.extractall(path=model_filename)

    return model_filename


@op("BioNeMo > Infer")
@mem.cache(verbose=1)
def infer(
    dataset_path: str, model_path: str | None = None, *, results_path: str
) -> str:
    """Infer on a dataset."""
    # This import is slow, so we only import it when we need it.
    from bionemo.geneformer.scripts.infer_geneformer import infer_model

    infer_model(
        data_path=dataset_path,
        checkpoint_path=model_path,
        results_path=DATA_PATH / results_path,
        include_hiddens=False,
        micro_batch_size=32,
        include_embeddings=True,
        include_logits=False,
        seq_length=2048,
        precision="bf16-mixed",
        devices=1,
        num_nodes=1,
        num_dataset_workers=10,
    )
    return DATA_PATH / results_path


@op("BioNeMo > Load results")
def load_results(results_path: str):
    embeddings = (
        torch.load(f"{results_path}/predictions__rank_0.pt")["embeddings"]
        .float()
        .cpu()
        .numpy()
    )
    return embeddings


@op("BioNeMo > Get labels")
def get_labels(adata):
    infer_metadata = adata.obs
    labels = infer_metadata["cell_type"].values
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    label_encoder.integer_labels = integer_labels
    return label_encoder


@op("BioNeMo > Plot labels", view="visualization")
def plot_labels(adata):
    infer_metadata = adata.obs
    labels = infer_metadata["cell_type"].values
    label_counts = Counter(labels)
    labels = list(label_counts.keys())
    values = list(label_counts.values())

    options = {
        "title": {
            "text": "Cell type counts for classification dataset",
            "left": "center",
        },
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"rotate": 45, "align": "right"},
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "name": "Count",
                "type": "bar",
                "data": values,
                "itemStyle": {"color": "#4285F4"},
            }
        ],
    }
    return options


@op("BioNeMo > Run benchmark")
@mem.cache(verbose=1)
def run_benchmark(data, labels, *, use_pca: bool = False):
    """
    data - contains the single cell expression (or whatever feature) in each row.
    labels - contains the string label for each cell

    data_shape (R, C)
    labels_shape (R,)
    """
    np.random.seed(1337)
    # Define the target dimension 'n_components'
    n_components = 10  # for example, adjust based on your specific needs

    # Create a pipeline that includes Gaussian random projection and RandomForestClassifier
    if use_pca:
        pipeline = Pipeline(
            [
                ("projection", PCA(n_components=n_components)),
                ("classifier", RandomForestClassifier(class_weight="balanced")),
            ]
        )
    else:
        pipeline = Pipeline(
            [("classifier", RandomForestClassifier(class_weight="balanced"))]
        )

    # Set up StratifiedKFold to ensure each fold reflects the overall distribution of labels
    cv = StratifiedKFold(n_splits=5)

    # Define the scoring functions
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(
            precision_score, average="macro"
        ),  # 'macro' averages over classes
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro"),
        # 'roc_auc' requires probability or decision function; hence use multi_class if applicable
        "roc_auc": make_scorer(roc_auc_score, multi_class="ovr"),
    }
    labels = labels.integer_labels
    # Perform stratified cross-validation with multiple metrics using the pipeline
    results = cross_validate(
        pipeline, data, labels, cv=cv, scoring=scoring, return_train_score=False
    )

    # Print the cross-validation results
    print("Cross-validation metrics:")
    results_out = {}
    for metric, scores in results.items():
        if metric.startswith("test_"):
            results_out[metric] = (scores.mean(), scores.std())
            print(f"{metric[5:]}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    predictions = cross_val_predict(pipeline, data, labels, cv=cv)

    # v Return confusion matrix and metrics.
    conf_matrix = confusion_matrix(labels, predictions)

    return results_out, conf_matrix


@op("BioNeMo > Plot confusion matrix", view="visualization")
@mem.cache(verbose=1)
def plot_confusion_matrix(benchmark_output, labels):
    cm = benchmark_output[1]
    labels = labels.classes_
    str_labels = [str(label) for label in labels]
    norm_cm = [[float(val / sum(row)) if sum(row) else 0 for val in row] for row in cm]
    # heatmap has the 0,0 at the bottom left corner
    num_rows = len(str_labels)
    heatmap_data = [
        [j, num_rows - i - 1, norm_cm[i][j]]
        for i in range(len(labels))
        for j in range(len(labels))
    ]

    options = {
        "title": {"text": "Confusion Matrix", "left": "center"},
        "tooltip": {"position": "top"},
        "xAxis": {
            "type": "category",
            "data": str_labels,
            "splitArea": {"show": True},
            "axisLabel": {"rotate": 70, "align": "right"},
        },
        "yAxis": {
            "type": "category",
            "data": list(reversed(str_labels)),
            "splitArea": {"show": True},
        },
        "grid": {
            "height": "70%",
            "width": "70%",
            "left": "20%",
            "right": "10%",
            "bottom": "10%",
            "top": "10%",
        },
        "visualMap": {
            "min": 0,
            "max": 1,
            "calculable": True,
            "orient": "vertical",
            "right": 10,
            "top": "center",
            "inRange": {
                "color": ["#E0F7FA", "#81D4FA", "#29B6F6", "#0288D1", "#01579B"]
            },
        },
        "series": [
            {
                "name": "Confusion matrix",
                "type": "heatmap",
                "data": heatmap_data,
                "emphasis": {"itemStyle": {"borderColor": "#333", "borderWidth": 1}},
                "itemStyle": {"borderColor": "#D3D3D3", "borderWidth": 2},
            }
        ],
    }
    return options


@op("BioNeMo > Plot accuracy comparison", view="visualization")
def accuracy_comparison(benchmark_output10m, benchmark_output100m):
    results_10m = benchmark_output10m[0]
    results_106M = benchmark_output100m[0]
    data = {
        "model": ["10M parameters", "106M parameters"],
        "accuracy_mean": [
            results_10m["test_accuracy"][0],
            results_106M["test_accuracy"][0],
        ],
        "accuracy_std": [
            results_10m["test_accuracy"][1],
            results_106M["test_accuracy"][1],
        ],
    }

    labels = data["model"]  # X-axis labels
    values = data["accuracy_mean"]  # Y-axis values
    error_bars = data["accuracy_std"]  # Standard deviation for error bars

    options = {
        "title": {
            "text": "Accuracy Comparison",
            "left": "center",
            "textStyle": {
                "fontSize": 20,  # Bigger font for title
                "fontWeight": "bold",  # Make title bold
            },
        },
        "grid": {
            "height": "70%",
            "width": "70%",
            "left": "20%",
            "right": "10%",
            "bottom": "10%",
            "top": "10%",
        },
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {
                "rotate": 45,  # Rotate labels for better readability
                "align": "right",
                "textStyle": {
                    "fontSize": 14,  # Bigger font for X-axis labels
                    "fontWeight": "bold",
                },
            },
        },
        "yAxis": {
            "type": "value",
            "name": "Accuracy",
            "min": 0,
            "max": 1,
            "interval": 0.1,  # Matches np.arange(0, 1.05, 0.05)
            "axisLabel": {
                "textStyle": {
                    "fontSize": 14,  # Bigger font for X-axis labels
                    "fontWeight": "bold",
                }
            },
        },
        "series": [
            {
                "name": "Accuracy",
                "type": "bar",
                "data": values,
                "itemStyle": {
                    "color": "#440154"  # Viridis color palette (dark purple)
                },
            },
            {
                "name": "Error Bars",
                "type": "errorbar",
                "data": [
                    [val - err, val + err] for val, err in zip(values, error_bars)
                ],
                "itemStyle": {"color": "#1f77b4"},
            },
        ],
    }
    return options


@op("BioNeMo > Plot f1 comparison", view="visualization")
def f1_comparison(benchmark_output10m, benchmark_output100m):
    results_10m = benchmark_output10m[0]
    results_106M = benchmark_output100m[0]
    data = {
        "model": ["10M parameters", "106M parameters"],
        "f1_score_mean": [
            results_10m["test_f1_score"][0],
            results_106M["test_f1_score"][0],
        ],
        "f1_score_std": [
            results_10m["test_f1_score"][1],
            results_106M["test_f1_score"][1],
        ],
    }

    labels = data["model"]  # X-axis labels
    values = data["f1_score_mean"]  # Y-axis values
    error_bars = data["f1_score_std"]  # Standard deviation for error bars

    options = {
        "title": {
            "text": "F1 Score Comparison",
            "left": "center",
            "textStyle": {
                "fontSize": 20,  # Bigger font for title
                "fontWeight": "bold",  # Make title bold
            },
        },
        "grid": {
            "height": "70%",
            "width": "70%",
            "left": "20%",
            "right": "10%",
            "bottom": "10%",
            "top": "10%",
        },
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {
                "rotate": 45,  # Rotate labels for better readability
                "align": "right",
                "textStyle": {
                    "fontSize": 14,  # Bigger font for X-axis labels
                    "fontWeight": "bold",
                },
            },
        },
        "yAxis": {
            "type": "value",
            "name": "F1 Score",
            "min": 0,
            "max": 1,
            "interval": 0.1,  # Matches np.arange(0, 1.05, 0.05),
            "axisLabel": {
                "textStyle": {
                    "fontSize": 14,  # Bigger font for X-axis labels
                    "fontWeight": "bold",
                }
            },
        },
        "series": [
            {
                "name": "F1 Score",
                "type": "bar",
                "data": values,
                "itemStyle": {
                    "color": "#440154"  # Viridis color palette (dark purple)
                },
            },
            {
                "name": "Error Bars",
                "type": "errorbar",
                "data": [
                    [val - err, val + err] for val, err in zip(values, error_bars)
                ],
                "itemStyle": {"color": "#1f77b4"},
            },
        ],
    }
    return options
