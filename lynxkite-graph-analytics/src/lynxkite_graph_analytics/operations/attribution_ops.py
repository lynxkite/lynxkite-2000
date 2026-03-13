"""Attribution demo operations to integrate GAT attribution visualizations into LynxKite."""

from lynxkite_core import ops

from .. import core
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..models.gat import SimpleGAT

op = ops.op_registration(core.ENV)


@op("Calculate GAT attribution", icon="graph", color="orange")
def gat_attribution_demo(graph: core.Bundle):
    """Run a minimal GAT forward pass and show attribution comparisons.

    This operation extracts numeric node features (or falls back to degree),
    runs a tiny GAT from the bundled graph, and uses the demo plotting
    utilities to render node/edge attributions.
    """

    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required to run the GAT attribution demo: " + str(e))

    # Extract node and edge tables
    nodes = graph.dfs["nodes"].copy()
    edges = graph.dfs["edges"].copy()

    if "source" not in edges.columns or "target" not in edges.columns:
        raise ValueError("Edges table must contain 'source' and 'target' columns.")

    # Identify numeric feature columns (ignore common layout/id cols)
    exclude = {"id", "x", "y", "lat", "long"}
    feature_cols = [
        c for c in nodes.columns if c not in exclude and is_numeric_dtype(nodes[c].dtype)
    ]

    # Build node feature tensor (fallback to simple degree if no features)
    if feature_cols:
        x = torch.tensor(nodes[feature_cols].fillna(0).to_numpy(dtype=float), dtype=torch.float32)
    else:
        # Degree fallback
        deg = edges.groupby("source").size().reindex(nodes.index, fill_value=0).astype(float)
        x = torch.tensor(deg.to_numpy().reshape(-1, 1), dtype=torch.float32)

    # Build edge_index tensor
    edge_index = torch.tensor(edges[["source", "target"]].to_numpy().T, dtype=torch.long)

    # Build minimal PyG Data object if available
    try:
        from torch_geometric.data import Data

        data = Data(x=x, edge_index=edge_index)
    except Exception:
        # If torch_geometric is not available, still attempt to run the SimpleGAT
        class SimpleData:
            def __init__(self, x, edge_index):
                self.x = x
                self.edge_index = edge_index

        data = SimpleData(x, edge_index)

    # Instantiate a tiny GAT and run a forward pass to extract attention
    model = SimpleGAT(in_channels=x.shape[1], hidden_channels=8, out_channels=1)
    model.eval()
    with torch.no_grad():
        try:
            out, attention = model(data, return_attention=True)
        except TypeError:
            # Some wrappers may return only (out, (edge_index, attn)) - normalize
            res = model(data)
            if isinstance(res, tuple) and len(res) == 2:
                out, attention = res
            else:
                out = res
                attention = None

    # Prepare simple node attribution: feature norm
    try:
        feature_norm = torch.from_numpy(np.linalg.norm(x.cpu().numpy(), axis=1))
    except Exception:
        feature_norm = torch.ones(x.shape[0])

    # Pull attention tensor if present
    attention_tensor = None
    if isinstance(attention, dict):
        # Prefer final layer attention if available
        if "layer2" in attention:
            attention_tensor = attention["layer2"]
        elif "alpha" in attention:
            attention_tensor = attention["alpha"]
        else:
            # If attention is given as (edge_index, attn)
            try:
                attention_tensor = list(attention.values())[0]
            except Exception:
                attention_tensor = None
    elif isinstance(attention, tuple) and len(attention) >= 2:
        # (edge_index, attn)
        attention_tensor = attention[1]

    # Attach computed values to the bundle so other boxes can consume them.
    # Copy the bundle to avoid modifying caller's object unexpectedly.
    out_bundle = graph.copy()

    # Ensure nodes and edges exist
    nodes_df = out_bundle.dfs.get("nodes")
    edges_df = out_bundle.dfs.get("edges")
    if nodes_df is None or edges_df is None:
        raise ValueError(
            "Bundle must contain 'nodes' and 'edges' tables to run GAT attribution demo."
        )

    # Add node-level attributes
    try:
        nodes_df = nodes_df.copy()
        nodes_df["feat_norm"] = feature_norm.cpu().numpy()
    except Exception:
        nodes_df["feat_norm"] = 0.0

    # Prediction confidence: try softmax for multi-class, sigmoid for single output
    pred_conf = None
    try:
        if out is not None:
            if hasattr(out, "detach"):
                o = out.detach()
            else:
                o = torch.tensor(out)
            if o.ndim == 2 and o.shape[1] > 1:
                probs = torch.softmax(o, dim=1)
                # confidence as max class probability
                pred_conf = probs.max(dim=1).values.cpu().numpy()
            else:
                # single logit per node
                pred_conf = torch.sigmoid(o.squeeze()).cpu().numpy()
        else:
            pred_conf = np.ones(len(nodes_df)) * 0.5
    except Exception:
        pred_conf = np.ones(len(nodes_df)) * 0.5

    nodes_df["pred_conf"] = pred_conf
    # degree
    try:
        deg = edges_df.groupby("source").size().reindex(nodes_df.index, fill_value=0).astype(float)
    except Exception:
        deg = pd.Series([0.0] * len(nodes_df), index=nodes_df.index)
    nodes_df["degree"] = deg.values

    # Attach back
    out_bundle.dfs["nodes"] = nodes_df

    # Add edge-level attention
    if attention_tensor is not None:
        try:
            attn_np = attention_tensor.detach().cpu().numpy()
            # If per-head, average heads
            if attn_np.ndim > 1:
                attn_vals = attn_np.mean(axis=1)
            else:
                attn_vals = attn_np
        except Exception:
            attn_vals = np.zeros(len(edges_df))
    else:
        attn_vals = np.zeros(len(edges_df))

    # Ensure same length as edges
    if len(attn_vals) < len(edges_df):
        attn_vals = np.pad(
            attn_vals, (0, max(0, len(edges_df) - len(attn_vals))), constant_values=0.0
        )
    edges_df = edges_df.copy()
    edges_df["attention"] = attn_vals[: len(edges_df)]
    out_bundle.dfs["edges"] = edges_df

    # Also store raw attention tensor in bundle.other for advanced use
    out_bundle.other["attention_tensor"] = attention_tensor

    return out_bundle
