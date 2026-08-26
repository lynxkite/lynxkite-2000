"""Custom operations for the Age prediction demo workspace."""

import numpy as np
from matplotlib import pyplot as plt
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


@op("Scatter plot", view="matplotlib", color="blue")
def scatter_plot(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    x_column: core.ColumnNameByTableName,
    y_column: core.ColumnNameByTableName,
    origin: bool = True,
    identity_line: bool = False,
):
    """
    Creates a scatter plot of the specified x and y columns from the given table.
    :param bundle: the bundle.
    :param table_name: the name of the table containing the data.
    :param x_column: the name of the column to be used for the x-axis.
    :param y_column: the name of the column to be used for the y-axis.
    :param origin: if True, the plot will include the origin (0,0) in the axes limits.
    :param identity_line: if True, an identity line will be drawn on the plot.
    """
    df = bundle.dfs[table_name].copy()
    x = np.stack(df[x_column].to_numpy())
    y = np.stack(df[y_column].to_numpy())

    fig, ax = plt.subplots(tight_layout=True)

    size = 200 / len(x) ** 0.5
    ax.scatter(x, y, s=size)

    if origin:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.set_xlim(min(0, x_lim[0]), max(0, x_lim[1]))
        ax.set_ylim(min(0, y_lim[0]), max(0, y_lim[1]))

    if identity_line:
        ax.axline((0, 0), slope=1, color="black", alpha=0.2)
    plt.xlabel(x_column)
    plt.ylabel(y_column)


@op("Rename columns", color="orange", icon="writing")
def rename_columns(
    b: core.Bundle, *, table_name: core.TableName, pairs: core.DropdownTextAdderByTableName
) -> core.Bundle:
    """
    Renames columns in the specified table according to the provided pairs of old and new names.
    :param b: the bundle.
    :param table_name: the table containing the columns to be renamed.
    :param pairs: the list of pairs (old_name, new_name).
    :return:
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    for old_name, new_name in pairs:
        df.rename(columns={old_name: new_name}, inplace=True)
    b.dfs[table_name] = df
    return b
