"""Custom operations for the Age prediction demo workspace."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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


@op("Train decision tree regression model", icon="circles")
def train_decision_tree(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    feature_column: core.ColumnNameByTableName,
    label_column: core.ColumnNameByTableName,
    max_depth: int = 7,
    min_impurity_decrease: float = 0.0,
    min_samples_leaf: int = 1,
    seed: int = 42,
    model_name: str = "decision_tree",
) -> core.Bundle:
    """
    :param seed: seed for random number generator.
    :param min_samples_leaf: minimum number of samples required to be at a leaf node.
    :param min_impurity_decrease: minimum impurity decrease required to split a node.
    :param max_depth: maximum depth of the tree.
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
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_impurity_decrease=min_impurity_decrease,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    tree = tree.fit(x_train, y_train)
    b.other[model_name] = tree
    return b


class GNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, conv_op_name: str):
        super().__init__()
        self.convs = nn.ModuleList()
        conv_op = getattr(geom_nn, conv_op_name)

        if num_layers == 1:
            self.convs.append(conv_op(in_dim, 1))
        else:
            self.convs.append(conv_op(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(conv_op(hidden_dim, hidden_dim))
            self.convs.append(conv_op(hidden_dim, 1))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)


@op("Train GNN classifier model", icon="circles", slow=True)
def train_gnn_classifier(
    b: core.Bundle,
    *,
    train_table_name: core.TableName,
    test_table_name: core.TableName,
    table_name: core.TableName,
    relation_name: core.RelationName,
    feature_column: core.ColumnNameByTableName,
    label_column: core.ColumnNameByTableName,
    prediction_column: str = "age_prediction",
    model_name: str = "model",
    iterations: int = 5000,
    use_labels_as_input: bool = True,
    batch_size: int = 80,
    learning_rate: float = 0.01,
    hidden_size: int = 16,
    num_conv_layers: int = 2,
    conv_operator: str = "GCNConv",
    random_seed: int = 2085407568,
) -> core.Bundle:
    """Train a full-batch GCN regressor and write predictions into train/test tables.

    :param b: The input bundle.
    :param train_table_name: Table containing labeled training rows.
    :param test_table_name: Table containing rows to predict.
    :param table_name: Full node table used by the graph relation.
    :param relation_name: Relation name that defines edge table and node keys.
    :param feature_column: Feature vector column in the full node table.
    :param label_column: Numeric label column in the training table.
    :param prediction_column: Output prediction column written to train/test tables.
    :param model_name: Key used to store the serialized trained model artifact in ``b.other``.
    :param iterations: Number of training iterations.
    :param use_labels_as_input: If true, append known train labels and a labeled-mask channel to node features.
    :param batch_size: Kept for workspace compatibility; not used in full-batch training.
    :param learning_rate: Adam learning rate.
    :param hidden_size: Hidden size in graph convolution layers.
    :param num_conv_layers: Number of graph convolution layers.
    :param conv_operator: Name of the torch_geometric convolution operator class.
    :param random_seed: Random seed for reproducibility.
    """
    b = b.copy()

    train_df = b.dfs[train_table_name].copy()
    test_df = b.dfs[test_table_name].copy()
    rel = next((r for r in b.relations if r.name == relation_name))
    node_df = b.dfs[table_name].copy()
    node_key = rel.source_key
    edge_df = b.dfs[rel.df].copy()

    node_ids = node_df[node_key].to_numpy()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    src_mapped = edge_df[rel.source_column].map(node_to_idx)
    tgt_mapped = edge_df[rel.target_column].map(node_to_idx)
    valid_edges = src_mapped.notna() & tgt_mapped.notna()

    source_indices = src_mapped[valid_edges].astype(int).to_numpy()
    target_indices = tgt_mapped[valid_edges].astype(int).to_numpy()

    undirected_src = np.concatenate([source_indices, target_indices])
    undirected_tgt = np.concatenate([target_indices, source_indices])
    edge_index = torch.tensor(np.vstack([undirected_src, undirected_tgt]), dtype=torch.long)

    features = np.asarray(node_df[feature_column].tolist(), dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    torch.manual_seed(random_seed)

    train_idx = train_df[node_key].map(node_to_idx)
    valid_train = train_idx.notna() & train_df[label_column].notna()

    train_idx = train_idx[valid_train].astype(int).to_numpy()
    y_train_raw = np.asarray(train_df.loc[valid_train, label_column].to_numpy(), dtype=float)

    y_mean, y_std = y_train_raw.mean(), y_train_raw.std() + 1e-6
    y_train_norm = (y_train_raw - y_mean) / y_std

    n_nodes = features.shape[0]

    if use_labels_as_input:
        label_channel = np.zeros((n_nodes, 1), dtype=float)
        mask_channel = np.zeros((n_nodes, 1), dtype=float)

        label_channel[train_idx, 0] = y_train_norm
        mask_channel[train_idx, 0] = 1.0

        features = np.hstack([features, label_channel, mask_channel])

    x_full = torch.tensor(features, dtype=torch.float32)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True

    y_all = torch.zeros(n_nodes, dtype=torch.float32)
    y_all[train_mask] = torch.tensor(y_train_norm, dtype=torch.float32)

    model = GNNRegressor(
        in_dim=x_full.shape[1],
        hidden_dim=hidden_size,
        num_layers=num_conv_layers,
        conv_op_name=conv_operator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(iterations):
        optimizer.zero_grad()
        out = model(x_full, edge_index)
        loss = F.mse_loss(out[train_mask], y_all[train_mask])

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_norm = model(x_full, edge_index).cpu().numpy()
        pred_values_all = (pred_norm * y_std) + y_mean

    pred_series = pd.Series(pred_values_all, index=node_ids)
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[prediction_column] = train_out[node_key].map(pred_series)
    test_out[prediction_column] = test_out[node_key].map(pred_series)

    b.dfs[train_table_name] = train_out
    b.dfs[test_table_name] = test_out

    b.other[model_name] = {
        "model_type": "GNNRegressor",
        "conv_operator": conv_operator,
        "in_dim": int(x_full.shape[1]),
        "hidden_size": int(hidden_size),
        "num_conv_layers": int(num_conv_layers),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "node_key": node_key,
        "prediction_column": prediction_column,
        "use_labels_as_input": bool(use_labels_as_input),
    }
    return b


@op("Numeric ids", color="green", icon="table-filled")
def numeric_id(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
):
    """replaces the ids in source/target tables, split tables, and edge table with global numeric ids"""
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
