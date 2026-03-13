"""GAT model utilities moved from the attribution demo.

This module contains a minimal GAT wrapper and a small model loader used
by the attribution visualizations.
"""

import pickle
from pathlib import Path
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PyGGATConv


class GATConv(nn.Module):
    """Wrapper around PyG's GATConv that extracts attention weights."""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        **kwargs,
    ):
        super(GATConv, self).__init__()

        self.gat = PyGGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
            add_self_loops=True,
            **kwargs,
        )

    def forward(self, x, edge_index, return_attention_weights=True):
        """Forward pass with optional attention weight return."""
        result = self.gat(x, edge_index, return_attention_weights=return_attention_weights)

        if return_attention_weights:
            out, (edge_index_with_loops, attention_weights) = result
            return out, (edge_index_with_loops, attention_weights)
        else:
            if isinstance(result, tuple):
                return result[0]
            return result


class SimpleGAT(nn.Module):
    """Simple 2-layer GAT model for demonstration."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(SimpleGAT, self).__init__()

        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True
        )
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False
        )
        self.dropout = dropout

    def forward(self, data, return_attention=True):
        """
        Forward pass through GAT layers.

        Args:
            data: PyG Data object with x and edge_index
            return_attention: Whether to return attention weights

        Returns:
            out: Output logits
            attention_weights: Dictionary with attention weights (if return_attention=True)
        """
        x, edge_index = data.x, data.edge_index

        # First layer
        if return_attention:
            result1 = self.conv1(x, edge_index, return_attention_weights=True)
            x, (edge_index_with_loops, alpha1) = result1
        else:
            x = self.conv1(x, edge_index, return_attention_weights=False)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer
        if return_attention:
            result2 = self.conv2(x, edge_index, return_attention_weights=True)
            x, (edge_index_with_loops2, alpha2) = result2
            attention_weights = {
                "edge_index": edge_index_with_loops2,
                "layer1": alpha1,
                "layer2": alpha2,
            }
            return x, attention_weights
        else:
            x = self.conv2(x, edge_index, return_attention_weights=False)
            return x


class ModelLoader:
    """Utility class for loading feature names."""

    def load_feature_names(self, dataset: str = "hbec") -> List[str]:
        """
        Load gene/feature names from pickle file.

        Args:
            dataset: Dataset name ('hbec' or 'liao')

        Returns:
            List of feature names
        """
        dataset_path = Path(__file__).parent / "dataset" / f"{dataset}_feat_names.pkl"

        if not dataset_path.exists():
            print(f"Warning: Feature names file not found at {dataset_path}")
            return []

        with open(dataset_path, "rb") as f:
            feature_names = pickle.load(f)

        print(f"Loaded {len(feature_names)} feature names from {dataset}")
        return feature_names
