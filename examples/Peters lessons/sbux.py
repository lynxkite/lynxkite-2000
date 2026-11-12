"""Custom operations for the Sbux demo workspace."""

from sklearn.linear_model import LogisticRegression

import numpy as np
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core


op = op_registration(core.ENV, "Sbux")


@op("Train logistic regression model", icon="circles")
def train_logreg(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    feature_column: core.ColumnNameByTableName,
    label_column: core.ColumnNameByTableName,
    model_name: str = "logistic",
    C: float = 1.0,
    l1_ratio: float = 0.0,
    solver: str = "lbfgs",
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> core.Bundle:
    """
    :param b: The bundle.
    :param table_name: The name of the table containing the training data.
    :param feature_column: The name of the column containing the feature vectors.
    :param label_column: The name of the column containing the labels.
    :param model_name: The name to assign to the trained model.
    :param C: Inverse of regularization strength; must be a positive float.
    :param l1_ratio: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
    :param solver: Algorithm to use in the optimization problem.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for stopping criteria.
    :param random_state: Seed for random number generator.
    """
    b = b.copy()
    train_df = b.dfs[table_name].copy()
    x_train = np.array(train_df[feature_column].tolist())
    y_train = train_df[label_column].to_numpy()

    logistic = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    ).fit(x_train, y_train)
    b.other[model_name] = logistic
    return b
