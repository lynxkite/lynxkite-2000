"""Custom operations for the Age prediction demo workspace."""

import numpy as np
from sklearn.linear_model import LinearRegression

from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "Age prediction")


@op("Train linear regression model", icon="circles")
def train_linreg(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    feature_column: core.ColumnNameByTableName,
    label_column: core.ColumnNameByTableName,
    model_name: str = "linear",
) -> core.Bundle:
    """
    :param b: The bundle.
    :param table_name: The name of the table containing the training data.
    :param feature_column: The name of the column containing the feature vectors.
    :param label_column: The name of the column containing the labels.
    :param model_name: The name to assign to the trained model.
    """
    b = b.copy()
    train_df = b.dfs[table_name].copy()
    x_train = np.array(train_df[feature_column].tolist())
    y_train = train_df[label_column].to_numpy()

    linear = LinearRegression().fit(x_train, y_train)
    b.other[model_name] = linear
    return b
