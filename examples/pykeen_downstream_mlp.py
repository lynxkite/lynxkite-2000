from lynxkite_core import ops
from lynxkite_graph_analytics import core

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


op = ops.op_registration("LynxKite Graph Analytics")


@op("Train/test/validation split")
def train_test_split(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed=1234,
):
    """Splits a dataframe in the bundle into separate "_train", "_test" and "_val" dataframes."""
    df = bundle.dfs[table_name]

    # Sample test data but keep the original index
    test = df.sample(frac=test_ratio, random_state=seed)

    # Drop using the original sampled indices
    remaining = df.drop(test.index)

    # Sample validation data from remaining data
    val = remaining.sample(frac=val_ratio / (1 - test_ratio), random_state=seed + 1)

    # The rest is training data
    train = remaining.drop(val.index).reset_index(drop=True)

    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    bundle = bundle.copy()
    bundle.dfs[f"{table_name}_train"] = train
    bundle.dfs[f"{table_name}_test"] = test
    bundle.dfs[f"{table_name}_val"] = val
    return bundle


@op("Merge features with embeddings")
def merge_feature_with_embedding(
    embeddings: core.Bundle,
    features: core.Bundle,
    *,
    embeddings_table: core.TableName,
    features_table: core.TableName,
    on_column: str,
):
    features.dfs[features_table][on_column] = features.dfs[features_table][on_column].astype(str)
    combined_df = features.dfs[features_table].merge(embeddings.dfs[embeddings_table], on=on_column)
    bundle = embeddings.copy()
    bundle.dfs["combined"] = combined_df
    return bundle


@op("Generate positive and negative samples")
def gen_pos_and_neg_sample(
    bundle: core.Bundle,
    *,
    edges_table: core.TableName,
    features_and_embeddings_table: core.TableName,
    seed: int = 1234,
):
    edges_df = bundle.dfs[edges_table]
    combined_df = bundle.dfs[features_and_embeddings_table]

    positive_samples = edges_df.merge(
        combined_df, left_on="head", right_on="node_label", suffixes=("_head", None)
    )
    positive_samples = positive_samples.drop(columns=["node_label"])
    positive_samples = positive_samples.merge(
        combined_df, left_on="tail", right_on="node_label", suffixes=("_tail", None)
    )
    positive_samples = positive_samples.drop(columns=["node_label"])
    positive_samples["label"] = [[1] for _ in range(len(positive_samples))]

    # Generate negative samples
    all_nodes = combined_df["node_label"].unique()
    positive_pairs = set(tuple(sorted(pair)) for pair in edges_df[["head", "tail"]].to_numpy())

    negative_samples = []
    num_positive_samples = len(positive_samples)
    num_negative_samples = 0

    # Randomly sample pairs until we have a comparable number of negative samples
    while num_negative_samples < num_positive_samples:
        source_node = np.random.choice(all_nodes)
        target_node = np.random.choice(all_nodes)

        # Ensure source and target are different and the pair is not in positive_pairs
        if (
            source_node != target_node
            and tuple(sorted((source_node, target_node))) not in positive_pairs
        ):
            negative_samples.append({"head": source_node, "tail": target_node})
            num_negative_samples += 1

    negative_samples_df = pd.DataFrame(negative_samples)

    # Merge with combined_df to get features and embeddings for negative samples
    negative_samples_df = negative_samples_df.merge(
        combined_df, left_on="head", right_on="node_label", suffixes=("_head", None)
    )
    negative_samples_df = negative_samples_df.drop(columns=["node_label"])
    negative_samples_df = negative_samples_df.merge(
        combined_df, left_on="tail", right_on="node_label", suffixes=("_tail", None)
    )
    negative_samples_df = negative_samples_df.drop(columns=["node_label"])
    negative_samples_df["label"] = [[0] for _ in range(len(negative_samples_df))]

    # Concatenate the positive and negative samples
    final_dataset = pd.concat([positive_samples, negative_samples_df], ignore_index=True)

    #  Shuffle the final dataset
    final_dataset = final_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    bundle.dfs["mlp_dataset"] = final_dataset
    return bundle


@op("Model evaluation")
def model_eval_(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    labels_column: core.ColumnNameByTableName,
    predictions_column: core.ColumnNameByTableName,
):
    bundle = bundle.copy()

    df = bundle.dfs[table_name]
    # Extract label values directly
    all_labels = df[labels_column].apply(lambda x: x[0] if isinstance(x, list) else x).values
    # Extract prediction values and convert to binary predictions
    all_predictions = (
        df[predictions_column].apply(lambda x: x[0] if isinstance(x, list) else x).values
    )
    all_predictions = (all_predictions > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    metrics_df = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1 score"],
            "score": [accuracy, precision, recall, f1],
        }
    )
    bundle.dfs["metrics"] = metrics_df

    return bundle


@op("Permute and corrupt data")
def permute_and_corrupt_data(
    bundle: core.Bundle, *, table_name: core.TableName, permute: str, seed: int
):
    """Creates permuted datasets"""
    bundle = bundle.copy()
    original_dataframe = bundle.dfs[table_name]
    permuted_df = original_dataframe.copy()

    # Identify original feature columns (those starting with 'feature_')
    original_feature_columns = [
        col
        for col in original_dataframe.columns
        if isinstance(col, str) and col.startswith("feature_")
    ]

    if permute == "features" or permute == "both":
        # Permute original feature columns
        for col in original_feature_columns:
            permuted_df[col] = permuted_df[col].sample(frac=1, random_state=seed).values
    if permute == "edges" or permute == "both":
        # Permute the 'label' column
        permuted_df["label"] = permuted_df["label"].sample(frac=1, random_state=seed + seed).values

    bundle.dfs["permuted_data"] = permuted_df
    return bundle
