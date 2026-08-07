"""Custom operations for the KNN demo workspace."""

from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core
from sklearn.datasets import fetch_openml

op = op_registration(core.ENV, "KNN")


@op("Fetch openml dataset")
def fetch_dataset(*, dataset_name: str, version: int = 1) -> core.Bundle:
    b = core.Bundle()
    dataset = fetch_openml(dataset_name, version=version, as_frame=True)
    b.dfs[dataset_name] = dataset.frame
    return b
