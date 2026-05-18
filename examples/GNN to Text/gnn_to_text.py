"""Custom operations for the "GNN to Text" example."""

import enum
import itertools
from lynxkite_core import ops
from lynxkite_graph_analytics import core, bundle
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
import click

if __name__ == "__main__":
    import joblib

    mem = joblib.Memory(".joblib-cache", verbose=False)
    ops.CACHE_WRAPPER = mem.cache


op = ops.op_registration("LynxKite Graph Analytics", "Attribution", icon="microscope")


def load_disease_gene_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    DIS_GENE_ROOT = root_path
    disease_dfs = []
    for file in os.listdir(DIS_GENE_ROOT):
        if file.endswith(("OT.tsv")):
            file_path = os.path.join(DIS_GENE_ROOT, file)
            # Derive disease name from the filename without extension
            disease = os.path.splitext(file)[0]
            # Read the file; assume it has a header with 'symbol' and 'globalScore'
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=0,  # use the file's header
                usecols=["symbol", "globalScore"],  # ensure we only read needed cols
            )
            # Add the disease column
            disease = disease.replace("_GENE_OT", "").replace("CR", "CRC")
            df["disease"] = disease
            print(f"Loaded {len(df)} gene associations for disease '{disease}' from {file}")
            disease_dfs.append(df)

    # Concatenate all disease DataFrames
    disease_df = pd.concat(disease_dfs, ignore_index=True)

    # Clean and filter
    disease_df["globalScore"] = pd.to_numeric(disease_df["globalScore"], errors="coerce")
    disease_df = (
        disease_df.dropna(subset=["globalScore"])
        .query("globalScore >= 0.5")
        .loc[:, ["symbol", "globalScore", "disease"]]
    )
    b.dfs["disease"] = disease_df
    return b


def load_drug_gene_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    b.dfs["drug_gene"] = pd.read_csv(f"{root_path}/drug_gene.csv")
    return b


def load_esm2_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    # RNA + ESM2 embeddings
    esm2_df = pd.read_parquet(f"{root_path}/esm2-gene-embeddings.parquet")
    gene2esm2 = dict(zip(esm2_df["genes"], esm2_df["embeddings"]))
    b.dfs["gene2esm2"] = pd.DataFrame(
        {"embedding": list(gene2esm2.values())}, index=list(gene2esm2.keys())
    )
    return b


def load_pct_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    clin_df = pd.read_excel(f"{root_path}/rna_data.xlsx", sheet_name="PCT raw data")
    clin_df = clin_df[
        clin_df["Tumor Type"].notna()  # Not NaN
        & (clin_df["Tumor Type"] != "GC")  # Tissue data is missing.
        & (clin_df["Tumor Type"].astype(str).str.strip() != "")  # Not empty
    ].copy()
    b.dfs["clin_df"] = clin_df
    b.dfs["meta"] = (
        clin_df[["Model", "Treatment", "Tumor Type"]].drop_duplicates().reset_index(drop=True)
    )
    return b


def index_genes(b: core.Bundle) -> core.Bundle:
    """Enumerates all the genes we have data for, and gives them sequential numbers."""
    root_path = b.other["root_path"]
    all_ppi_genes = set(
        itertools.chain.from_iterable(
            load_ppi_for_tissue(root_path, t)[["gene1", "gene2"]].values.flatten()
            for t in os.listdir(root_path + "/Tissue")
        )
    )
    esm2_genes = set(b.dfs["gene2esm2"].index)
    avail_genes = sorted(all_ppi_genes & esm2_genes)
    gene_to_idx = {g: i for i, g in enumerate(avail_genes)}
    b.dfs["gene_to_idx"] = pd.DataFrame(
        list(gene_to_idx.values()), index=list(gene_to_idx.keys()), columns=["idx"]
    )
    return b


def classify_mrecist(b: core.Bundle) -> core.Bundle:
    num_timestamps_in_past = 10
    clin_df = b.dfs["clin_df"]
    times_col = []
    vols_col = []
    seqs_col = []
    class_col = []
    for _, row in b.dfs["meta"].iterrows():
        pdx_id = row["Model"]
        tumor = row["Tumor Type"]
        drug = row["Treatment"]
        sub = clin_df[
            (clin_df["Model"] == pdx_id)
            & (clin_df["Tumor Type"] == tumor)
            & (clin_df["Treatment"] == drug)
        ].sort_values("Days Post T0")
        times = sub["Days Post T0"].values
        vols = sub["Volume (mm3)"].values
        norm = vols / vols[0]
        seqs_col.append(norm[:num_timestamps_in_past])
        future_vols = vols[num_timestamps_in_past:]
        vols_col.append(future_vols)
        times_col.append(times[num_timestamps_in_past:])
        baseline = vols[0]
        percent_growth = 100 * (future_vols - baseline) / baseline
        if len(percent_growth) == 0:
            class_col.append("unknown")
        else:
            class_col.append("good" if percent_growth.min() <= 35 else "bad")
    b = b.copy()
    b.dfs["meta"] = b.dfs["meta"].copy()
    b.dfs["meta"]["future_timestamps"] = pd.Series(times_col)
    b.dfs["meta"]["future_volumes"] = pd.Series(vols_col)
    b.dfs["meta"]["past_volumes"] = pd.Series(seqs_col)
    b.dfs["meta"]["mrecist_class"] = pd.Series(class_col)
    b.dfs["meta"] = b.dfs["meta"].query("future_volumes.str.len() > 2").reset_index(drop=True)
    return b


@ops.cached
def load_ppi_for_tissue(root_path: str, tissue: str) -> pd.DataFrame:
    path = os.path.join(root_path, "Tissue", tissue, "edges.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=["gene1", "gene2", "score"])


@ops.cached
def load(root_path):
    res = []
    res.append(load_disease_gene_data(root_path=root_path))
    res.append(load_drug_gene_data(root_path=root_path))
    res.append(load_esm2_data(root_path=root_path))
    pct = load_pct_data(root_path=root_path)
    pct = classify_mrecist(pct)
    res.append(pct)
    b = bundle.merge_bundles(res, merge_mode=bundle.BundleMergeMode.concatenate_non_unique)
    b = index_genes(b)
    b = classify_mrecist(b)
    return b


@ops.cached
def precompute_tissue_graphs(b: core.Bundle):
    """Precompute PPI edges, disease gene indices, and ESM2 features for each tumor type."""
    gene_to_idx = b.dfs["gene_to_idx"]["idx"].to_dict()
    avail_genes = b.dfs["gene_to_idx"].index
    gene2esm2 = b.dfs["gene2esm2"]["embedding"]
    esm2_mat = torch.stack([torch.tensor(gene2esm2[g], dtype=torch.float32) for g in avail_genes])
    root_path = b.other["root_path"]
    tumor_types = b.dfs["meta"]["Tumor Type"].unique()
    cache = {}
    for tumor in tumor_types:
        # PPI edges (gene-gene only, no drug/disease nodes)
        ppi_df = load_ppi_for_tissue(root_path, tumor)
        rows, cols = [], []
        for u, v in zip(ppi_df.gene1, ppi_df.gene2):
            if u in gene_to_idx and v in gene_to_idx:
                rows.append(gene_to_idx[u])
                cols.append(gene_to_idx[v])
        ppi_edges = torch.tensor([rows, cols], dtype=torch.long)
        ppi_edges = pyg_utils.to_undirected(ppi_edges)
        ppi_edges, _ = pyg_utils.add_self_loops(ppi_edges)
        # Disease-associated gene indices
        df_dis = b.dfs["disease"]
        dis = df_dis[df_dis["disease"] == tumor].symbol
        dis_idx = [gene_to_idx[g] for g in dis if g in gene_to_idx]
        cache[tumor] = {
            "ppi_edges": ppi_edges,
            "disease_gene_idx": torch.tensor(dis_idx, dtype=torch.long)
            if dis_idx
            else torch.empty(0, dtype=torch.long),
        }
    return esm2_mat, cache


def _get_drug_gene_idx(b: core.Bundle, drug: str):
    """Helper to get gene indices for a drug's targets."""
    gene_to_idx = b.dfs["gene_to_idx"]["idx"].to_dict()
    df = b.dfs["drug_gene"]
    dg = df[df["Treatment"] == drug]["Gene"]
    idx = [gene_to_idx[g] for g in dg if g in gene_to_idx]
    return torch.tensor(idx, dtype=torch.long) if idx else torch.empty(0, dtype=torch.long)


def make_graph(b: core.Bundle, batch_index: int, esm2_mat, tissue_cache):
    """Construct a PyG Data graph for the sample at batch_index in meta."""
    row = b.dfs["meta"].iloc[batch_index]
    tumor = row["Tumor Type"]
    drug = row["Treatment"]
    avail_genes = b.dfs["gene_to_idx"].index

    # PPI-only edges
    tc = tissue_cache[tumor]
    drug_gene_idx = _get_drug_gene_idx(b, drug)
    disease_gene_idx = tc["disease_gene_idx"]
    num_genes = len(avail_genes)
    drug_flag = torch.zeros(num_genes, 1, dtype=torch.float32)
    drug_flag[drug_gene_idx, 0] = 1.0
    disease_flag = torch.zeros(num_genes, 1, dtype=torch.float32)
    disease_flag[disease_gene_idx, 0] = 1.0
    x = torch.cat([esm2_mat, drug_flag, disease_flag], dim=1)
    return Data(
        x=x,
        edge_index=tc["ppi_edges"],
        drug_gene_idx=drug_gene_idx,
        disease_gene_idx=disease_gene_idx,
    )


def make_graph_from_strings(
    b: core.Bundle,
    drug: str,
    disease: str,
    esm2_mat=None,
    tissue_cache=None,
):
    """Build a PyG graph for a (drug, disease) pair without needing a meta row."""
    avail_genes = b.dfs["gene_to_idx"].index
    num_genes = len(avail_genes)

    if esm2_mat is None or tissue_cache is None:
        esm2_mat, tissue_cache = precompute_tissue_graphs(b)

    tc = tissue_cache[disease]
    drug_gene_idx = _get_drug_gene_idx(b, drug)
    disease_gene_idx = tc["disease_gene_idx"]
    drug_flag = torch.zeros(num_genes, 1, dtype=torch.float32)
    drug_flag[drug_gene_idx, 0] = 1.0
    disease_flag = torch.zeros(num_genes, 1, dtype=torch.float32)
    disease_flag[disease_gene_idx, 0] = 1.0
    x = torch.cat([esm2_mat, drug_flag, disease_flag], dim=1)
    return Data(
        x=x,
        edge_index=tc["ppi_edges"],
        drug_gene_idx=drug_gene_idx,
        disease_gene_idx=disease_gene_idx,
    )


class GATModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=2,
        dropout=0.2,
    ):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(in_channels)
        self.compressor = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ELU(),
        )
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        # Classify from pooled drug-target and disease-gene representations.
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x,
        edge_index,
        drug_gene_idx,
        disease_gene_idx,
        return_attention=False,
    ):
        x = self.norm(x)
        x = self.compressor(x)
        # Phase 1: GAT on PPI graph to get contextualized gene embeddings
        if return_attention:
            x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
            attn1 = None
        x = F.elu(x)
        x = self.dropout(x)
        if return_attention:
            x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
            attn2 = None
        x = F.elu(x)
        # Phase 2: Pool drug-target and disease-associated gene embeddings
        if drug_gene_idx.numel() > 0:
            drug_repr = x[drug_gene_idx].max(dim=0).values
        else:
            drug_repr = torch.zeros(x.size(1), device=x.device)
        if disease_gene_idx.numel() > 0:
            disease_repr = x[disease_gene_idx].max(dim=0).values
        else:
            disease_repr = torch.zeros(x.size(1), device=x.device)
        pooled = torch.cat([drug_repr, disease_repr], dim=0)
        logit = self.classifier(pooled).squeeze()
        if return_attention:
            return logit, attn1, attn2
        return logit


def save_model(
    model,
    path="gat_model.pt",
    optimizer=None,
    scaler=None,
    epoch=None,
):
    """Save the trained GAT model to disk."""
    ckpt = {
        "state_dict": model.state_dict(),
        "in_channels": model.norm.num_features,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    torch.save(ckpt, path)
    print(f"Model saved to {path}")


def load_model(path="gat_model.pt", device=None):
    """Load a saved GAT model from disk."""
    device = device or DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "num_drugs" in ckpt or "num_diseases" in ckpt:
        raise ValueError(
            "Checkpoint uses the old embedding-based architecture. "
            "Retrain a model with the current binary-feature architecture."
        )
    model = GATModel(in_channels=ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(
    model,
    b: core.Bundle,
    drug: str,
    disease: str,
    esm2_mat=None,
    tissue_cache=None,
) -> float:
    """Return the predicted probability that the drug is effective for the disease."""
    model.eval()
    graph = make_graph_from_strings(b, drug, disease, esm2_mat, tissue_cache)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logit = model(
            graph.x.to(DEVICE),
            graph.edge_index.to(DEVICE),
            graph.drug_gene_idx.to(DEVICE),
            graph.disease_gene_idx.to(DEVICE),
        )
    return torch.sigmoid(logit).item()


def predict_with_attribution(
    model,
    b: core.Bundle,
    drug: str,
    disease: str,
    esm2_mat=None,
    tissue_cache=None,
):
    """Return probability and per-node attention scores.

    Returns:
        prob: Predicted probability that the drug is effective.
        node_scores: DataFrame with gene names and their
                     aggregated attention scores across both GAT layers.
    """
    model.eval()
    graph = make_graph_from_strings(b, drug, disease, esm2_mat, tissue_cache)
    avail_genes = list(b.dfs["gene_to_idx"].index)
    node_names = avail_genes
    num_nodes = len(node_names)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logit, (ei1, aw1), (ei2, aw2) = model(
            graph.x.to(DEVICE),
            graph.edge_index.to(DEVICE),
            graph.drug_gene_idx.to(DEVICE),
            graph.disease_gene_idx.to(DEVICE),
            return_attention=True,
        )
    prob = torch.sigmoid(logit).item()

    # aw1/aw2 shape: (num_edges, heads)
    def aggregate_attention(edge_index, attn_weights):
        attn = attn_weights.mean(dim=1)
        attn[edge_index[0] == edge_index[1]] = 0  # Self-attention doesn't count.
        scores = torch.zeros(num_nodes, device=attn.device)
        for i in range(num_nodes):
            # The attention on the information that went out from the node. This reflects how much the node influenced others.
            outgoing = attn[edge_index[0] == i]
            scores[i] = outgoing.sum()
        return scores.cpu()

    scores1 = aggregate_attention(ei1, aw1)
    scores1[graph.drug_gene_idx] += 1.1
    scores1[graph.drug_gene_idx] *= 10
    scores1[graph.disease_gene_idx] *= 10
    scores1 /= scores1.max()
    scores2 = aggregate_attention(ei2, aw2)
    scores2[graph.drug_gene_idx] += 1.1
    scores2[graph.drug_gene_idx] *= 10
    scores2[graph.disease_gene_idx] *= 10
    scores2 /= scores2.max()
    combined = scores1 + scores2

    df = pd.DataFrame({"node": node_names, "attention_score": combined.numpy()})
    df = df.sort_values("attention_score", ascending=False).reset_index(drop=True)
    return prob, df


def save_histogram(attn_weights, filename):
    """Save a histogram of attention weights to disk."""
    import matplotlib.pyplot as plt

    attn = (
        attn_weights.mean(dim=1).cpu().numpy()
        if attn_weights.dim() > 1
        else attn_weights.cpu().numpy()
    )
    attn = attn[attn > 0]
    bins = 1 if attn.std() < 0.001 else 50
    print(f"std: {attn.std():.4f}, mean: {attn.mean():.4f}, max: {attn.max():.4f}, bins: {bins}")
    plt.figure(figsize=(6, 4))
    plt.hist(attn, bins=bins, color="blue", alpha=0.7)
    plt.yscale("log")
    plt.title("Attention Weights Distribution")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved attention histogram to {filename}")


def top_genes(attribution_df: pd.DataFrame, n: int = 10) -> list[str]:
    """Return the names of the top N most important genes from an attribution DataFrame."""
    genes = attribution_df[~attribution_df["node"].str.contains(":")]
    return genes.head(n)["node"].tolist()


@op("Load GAT model", icon="file-filled", color="green", slow=True)
async def load_model_op(*, model_path, root_path) -> core.Bundle:
    """Load the model from a checkpoint."""
    b = load(root_path)
    load_model(model_path)
    b.other["model"] = model_path
    return b


class TumorTypes(enum.StrEnum):
    BRCA = "BRCA"
    CM = "CM"
    CRC = "CRC"
    NSCLC = "NSCLC"
    PDAC = "PDAC"


class Drugs(enum.StrEnum):
    _5FU = "5FU"
    BGJ398 = "BGJ398"
    BKM120 = "BKM120"
    BKM120_LDE225 = "BKM120 + LDE225"
    BKM120_LJC049 = "BKM120 + LJC049"
    BKM120_binimetinib = "BKM120 + binimetinib"
    BKM120_encorafenib = "BKM120 + encorafenib"
    BYL719 = "BYL719"
    BYL719_HSP990 = "BYL719 + HSP990"
    BYL719_LEE011 = "BYL719 + LEE011"
    BYL719_LGH447 = "BYL719 + LGH447"
    BYL719_LJM716 = "BYL719 + LJM716"
    BYL719_binimetinib = "BYL719 + binimetinib"
    BYL719_cetuximab = "BYL719 + cetuximab"
    BYL719_cetuximab_encorafenib = "BYL719 + cetuximab + encorafenib"
    BYL719_encorafenib = "BYL719 + encorafenib"
    CGM097 = "CGM097"
    CKX620 = "CKX620"
    CLR457 = "CLR457"
    HDM201 = "HDM201"
    HSP990 = "HSP990"
    INC280 = "INC280"
    INC280_trastuzumab = "INC280 + trastuzumab"
    INC424 = "INC424"
    INC424_binimetinib = "INC424 + binimetinib"
    LCL161_paclitaxel = "LCL161 + paclitaxel"
    LDE225 = "LDE225"
    LDK378 = "LDK378"
    LEE011 = "LEE011"
    LEE011_binimetinib = "LEE011 + binimetinib"
    LEE011_encorafenib = "LEE011 + encorafenib"
    LEE011_everolimus = "LEE011 + everolimus"
    LFA102 = "LFA102"
    LFW527_binimetinib = "LFW527 + binimetinib"
    LFW527_everolimus = "LFW527 + everolimus"
    LGH447 = "LGH447"
    LGW813 = "LGW813"
    LJC049 = "LJC049"
    LJM716 = "LJM716"
    LJM716_trastuzumab = "LJM716 + trastuzumab"
    LKA136 = "LKA136"
    LLM871 = "LLM871"
    TAS266 = "TAS266"
    WNT974 = "WNT974"
    abraxane = "abraxane"
    binimetinib = "binimetinib"
    binimetinib_3_5mpk = "binimetinib-3.5mpk"
    cetuximab = "cetuximab"
    cetuximab_encorafenib = "cetuximab + encorafenib"
    dacarbazine = "dacarbazine"
    encorafenib = "encorafenib"
    encorafenib_binimetinib = "encorafenib + binimetinib"
    erlotinib = "erlotinib"
    everolimus = "everolimus"
    figitumumab = "figitumumab"
    figitumumab_binimetinib = "figitumumab + binimetinib"
    gemcitabine_50mpk = "gemcitabine-50mpk"
    paclitaxel = "paclitaxel"
    tamoxifen = "tamoxifen"
    trametinib = "trametinib"
    trastuzumab = "trastuzumab"
    untreated = "untreated"


@op("View prediction", color="blue", view="table_view", icon="eye")
async def view_prediction(b: core.Bundle) -> core.Bundle:
    b = bundle.Bundle(
        dfs={
            "prediction": pd.DataFrame(
                {
                    "drug": [b.other["prediction"]["drug"]],
                    "disease": [b.other["prediction"]["disease"]],
                    "prob": [b.other["prediction"]["prob"]],
                }
            )
        }
    )
    print(b)
    return b.to_table_view()


@op("Predict with attribution", icon="sparkles", color="orange", slow=True, cache=False)
async def predict_with_attribution_op(
    b: core.Bundle, *, drug: Drugs, disease: TumorTypes
) -> core.Bundle:
    """Run inference with attribution, and save the results."""
    esm2_mat, tissue_cache = precompute_tissue_graphs(b)
    model = load_model(b.other["model"])
    prob, attr_df = predict_with_attribution(model, b, drug, disease, esm2_mat, tissue_cache)
    b = b.copy()
    b.dfs["attribution"] = attr_df
    b.other["prediction"] = {"prob": prob, "drug": str(drug), "disease": str(disease)}
    return b


DISEASE_NAMES = {
    "BRCA": "breast cancer",
    "CM": "cutaneous melanoma",
    "CRC": "colorectal cancer",
    "NSCLC": "non-small cell lung cancer",
    "PDAC": "pancreatic ductal adenocarcinoma",
}


@op("Attribution to text", icon="message-circle-filled", color="blue", view="table_view")
def attribution_to_text(b: core.Bundle, *, top_n=3):
    """Generate an explanation based on the attribution data."""
    genes = top_genes(b.dfs["attribution"], n=top_n)
    prediction = b.other["prediction"]
    prob = prediction["prob"]
    drug = str(prediction["drug"])
    disease = str(prediction["disease"])
    disease = DISEASE_NAMES.get(disease, disease)  # Map to nicer name if available

    effective = "effective" if prob >= 0.5 else "not effective"
    lines = []
    lines.append("## Prediction")
    lines.append(f"**{drug} is {effective} against {disease}** ({prob:.0%} probability)")
    lines.append(f"Top {top_n} genes by attribution: {', '.join(genes)}")

    explanation = explain_prediction(drug, disease, prob, genes)
    lines.append("## Explanation")
    lines.append(explanation)
    return core.Bundle(
        dfs={"explanation": pd.DataFrame({"text": ["\n\n".join(lines)]})}
    ).to_table_view()


@op("Visualize attribution", view="visualization", icon="eye", color="blue")
def visualize_attribution(
    b: core.Bundle,
    *,
    top_n: int = 10,
    neighbors_per_node: int = 10,
) -> dict:
    """Visualize the attribution on the top N genes and their neighbors in the PPI network.

    Args:
        b: Bundle containing gene_to_idx and PPI data.
        top_n: Number of top attention nodes to include.
        neighbors_per_node: Number of top neighbors to include for each top node.

    Returns:
        ECharts configuration dict for graph visualization.
    """
    import matplotlib.cm
    import networkx as nx

    # Get top nodes by attention
    attribution_df = b.dfs["attribution"]
    top_nodes = attribution_df.head(top_n)["node"].tolist()
    attention_scores = dict(zip(attribution_df["node"], attribution_df["attention_score"]))

    # Load PPI edges for this tissue
    root_path = b.other["root_path"]
    disease = str(b.other["prediction"]["disease"])
    ppi_df = load_ppi_for_tissue(root_path, disease)
    gene_to_idx = set(b.dfs["gene_to_idx"].index)

    # Build adjacency from PPI - iterate over columns directly (much faster than iterrows)
    adjacency = {}
    for g1, g2 in zip(ppi_df["gene1"], ppi_df["gene2"]):
        if g1 in gene_to_idx and g2 in gene_to_idx:
            adjacency.setdefault(g1, []).append(g2)
            adjacency.setdefault(g2, []).append(g1)

    # Collect subgraph nodes: top nodes + their top neighbors
    subgraph_nodes = set(top_nodes)
    for node in top_nodes:
        if node in adjacency:
            # Sort neighbors by attention score, take top ones
            neighbors = adjacency[node]
            neighbors_sorted = sorted(
                neighbors, key=lambda n: attention_scores.get(n, 0), reverse=True
            )
            subgraph_nodes.update(neighbors_sorted[:neighbors_per_node])

    # Build edges for subgraph
    edges = []
    seen_edges = set()
    for node in subgraph_nodes:
        if node in adjacency:
            for neighbor in adjacency[node]:
                if neighbor in subgraph_nodes:
                    edge_key = tuple(sorted([node, neighbor]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append((node, neighbor))

    # Create NetworkX graph for layout
    G = nx.Graph()
    G.add_nodes_from(subgraph_nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, iterations=50, seed=42)

    # Color mapping based on attention scores
    scores = [attention_scores.get(n, 0) for n in subgraph_nodes]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score if max_score > min_score else 1

    cmap = matplotlib.cm.get_cmap("viridis")

    def score_to_color(score):
        normalized = (score - min_score) / score_range
        r, g, b, _ = cmap(normalized)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    # Build ECharts config
    node_list = list(subgraph_nodes)

    data = []
    for node in node_list:
        score = attention_scores.get(node, 0)
        x, y = pos[node]
        is_top_node = node in top_nodes
        data.append(
            {
                "id": node,
                "name": node,
                "x": float(x * 100),
                "y": float(y * 100),
                "symbolSize": 20 if is_top_node else 10,
                "itemStyle": {"color": score_to_color(score)},
                "label": {"show": is_top_node},
                "value": f"{score:.4f}",
            }
        )

    links = [{"source": e[0], "target": e[1]} for e in edges]

    return {
        "animationDuration": 500,
        "animationEasingUpdate": "quinticInOut",
        "tooltip": {"show": True, "formatter": "{b}: {c}"},
        "visualMap": {
            "min": min_score,
            "max": max_score,
            "text": ["High", "Low"],
            "realtime": False,
            "calculable": True,
            "inRange": {"color": ["#440154", "#21918c", "#fde725"]},  # viridis-like
        },
        "series": [
            {
                "type": "graph",
                "layout": "none",
                "roam": True,
                "lineStyle": {
                    "color": "#cccccc",
                    "curveness": 0.1,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {"width": 3},
                },
                "label": {"position": "top", "formatter": "{b}"},
                "data": data,
                "links": links,
            },
        ],
    }


@ops.cached
def explain_prediction(
    drug: str, disease: str, prob: float, gene_list: list[str], add_context: bool = True
) -> str:
    """Ask an OpenAI model to explain why the GNN made its prediction based on top genes."""
    effective = "effective" if prob >= 0.5 else "not effective"
    genes_str = ", ".join(gene_list)
    context = []

    def add_to_context(file_path):
        if os.path.exists(file_path):
            print(f"Adding context from {file_path}")
            with open(file_path, "r") as f:
                contents = f.read().strip()
                context.append(contents)
        else:
            print(f"Context file {file_path} not found, skipping.")

    if add_context:
        add_to_context(f"../wp_pages/{disease.replace(' ', '_')}.wiki")
        for gene in gene_list:
            add_to_context(f"../gene_pages/{gene}.wiki")
        for d in drug.split("+"):
            d = d.strip()
            add_to_context(f"../wp_pages/{d}.wiki")
            context.append(f"## Known targets of {d}:\n")
            drug_gene_path = "uploads/ode-gnn/drug_gene.csv"
            df = pd.read_csv(drug_gene_path)
            targets = df[df["Treatment"] == d]["Gene"].tolist()
            context.append(", ".join(targets) if targets else "None found.")
    prompt = "\n".join(context) + (
        f"\nA neural network predicts that {drug} is {effective} against {disease}, "
        f"after considering the role of genes {genes_str}. "
        f"What is the likely underlying biological explanation?"
    )
    return ask_ai(prompt)


@ops.cached
def ask_ai(prompt: str) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        # Set OPENAI_BASE_URL=http://localhost:8088/v1 to use a local model, such as Gemma 4.
        # No code change is needed.
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Your task is to generate explanations. The user is unable to respond. Do not suggest follow-up questions or ask for clarification. Just provide a concise explanation based on the prompt.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


EPOCHS = 3
LABEL_MAP = {"good": 1.0, "bad": 0.0}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--epochs", default=EPOCHS, help="Number of training epochs.")
@click.option("--samples", default=100, help="Number of samples to train on.")
@click.option("--output", default="gat_model.pt", help="Path to save the model.")
@click.option(
    "--continue", "continue_training", is_flag=True, help="Continue training from checkpoint."
)
def train(epochs, samples, output, continue_training):
    """Train the GAT model and save it."""
    b = load(root_path="uploads/ode-gnn")
    meta = b.dfs["meta"].head(samples)
    print(f"Using device: {DEVICE}")

    esm2_mat, tissue_cache = precompute_tissue_graphs(b)
    in_channels = esm2_mat.shape[1] + 2  # ESM2 + drug/disease binary flags

    start_epoch = 0
    if continue_training and os.path.exists(output):
        print(f"Loading checkpoint from {output}...")
        ckpt = torch.load(output, map_location=DEVICE, weights_only=False)
        if "num_drugs" in ckpt or "num_diseases" in ckpt:
            raise ValueError(
                "Checkpoint uses the old embedding-based architecture. "
                "Retrain a model with the current binary-feature architecture."
            )
        model = GATModel(in_channels=ckpt["in_channels"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler = torch.amp.GradScaler("cuda")
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch + 1}")
    else:
        model = GATModel(in_channels=in_channels).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda")

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        indices = np.random.permutation(len(meta))
        for i in tqdm(indices, desc=f"Epoch {epoch + 1}"):
            row = meta.iloc[i]
            label_str = row["mrecist_class"]
            if label_str not in LABEL_MAP:
                continue
            label = torch.tensor(LABEL_MAP[label_str], device=DEVICE)
            graph = make_graph(b, i, esm2_mat, tissue_cache)
            with torch.amp.autocast("cuda"):
                out = model(
                    graph.x.to(DEVICE),
                    graph.edge_index.to(DEVICE),
                    graph.drug_gene_idx.to(DEVICE),
                    graph.disease_gene_idx.to(DEVICE),
                )
                loss = loss_fn(out, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pred = (out.detach() > 0).float()
            correct += (pred == label).item()
            total += 1

        avg_loss = epoch_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        print(
            f"Epoch {epoch + 1}/{start_epoch + epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}"
        )

    save_model(model, output, optimizer=optimizer, scaler=scaler, epoch=epoch)


@cli.command()
@click.argument("drug")
@click.argument("disease")
@click.option("--model-path", default="gat_model.pt", help="Path to the saved model.")
@click.option("--top-n", default=3, help="Number of top genes to include in the explanation.")
def explain(drug, disease, model_path, top_n):
    """Load the model, run inference with attribution, and print an AI explanation."""
    b = load(root_path="uploads/ode-gnn")
    esm2_mat, tissue_cache = precompute_tissue_graphs(b)
    model = load_model(model_path)
    prob, attr_df = predict_with_attribution(model, b, drug, disease, esm2_mat, tissue_cache)
    genes = top_genes(attr_df, n=top_n)

    effective = "effective" if prob >= 0.5 else "not effective"
    print(f"Prediction:\n{drug} is {effective} against {disease} ({prob:.0%} probability)")
    print(f"Top {top_n} genes: {', '.join(genes)}")

    explanation = explain_prediction(drug, disease, prob, genes)
    print("\nExplanation:")
    print(explanation)


if __name__ == "__main__":
    cli()
