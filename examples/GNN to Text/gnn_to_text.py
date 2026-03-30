"""Custom operations for the "GNN to Text" example."""

import itertools
from lynxkite_core import ops
from lynxkite_graph_analytics import core
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import os.path
import pandas as pd
from tqdm import tqdm
import click

if __name__ == "__main__":
    import joblib

    mem = joblib.Memory(".joblib-cache")
    ops.CACHE_WRAPPER = mem.cache


pdx_op = ops.op_registration("LynxKite Graph Analytics", "PDX", icon="microscope")


@pdx_op("Load disease-gene data", slow=True, color="green")
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
            df["disease"] = disease
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


@pdx_op("Load drug-gene data", slow=True, color="green")
def load_drug_gene_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    drug_gene_df = pd.read_csv(f"{root_path}/DGIdb_interactions.tsv", sep="\t")[
        ["gene_name", "drug_name"]
    ].assign(drug_name=lambda d: d.drug_name.str.lower())
    b.dfs["drug_gene"] = drug_gene_df
    return b


@pdx_op("Load RNA data", slow=True, color="green")
def load_rna_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    rna_df = pd.read_excel(f"{root_path}/rna_data.xlsx", sheet_name="RNAseq_fpkm")
    b.dfs["rna_df"] = rna_df.set_index("Sample")
    return b


@pdx_op("Load ESM2 data", slow=True, color="green")
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


@pdx_op("Load PCT data", slow=True, color="green")
def load_pct_data(*, root_path: str) -> core.Bundle:
    b = core.Bundle()
    b.other["root_path"] = root_path
    clin_df = pd.read_excel(f"{root_path}/rna_data.xlsx", sheet_name="PCT raw data")
    clin_df = clin_df[
        clin_df["Tumor Type"].notna()  # Not NaN
        & (clin_df["Tumor Type"].astype(str).str.strip() != "")  # Not empty
    ].copy()
    b.dfs["clin_df"] = clin_df
    b.dfs["meta"] = (
        clin_df[["Model", "Treatment", "Tumor Type"]].drop_duplicates().reset_index(drop=True)
    )
    return b


@pdx_op("Index genes", slow=True)
def index_genes(b: core.Bundle) -> core.Bundle:
    """Enumerates all the genes we have data for, and gives them sequential numbers."""
    root_path = b.other["root_path"]
    all_rna_genes = set(b.dfs["rna_df"].index)
    all_ppi_genes = set(
        itertools.chain.from_iterable(
            load_ppi_for_tissue(root_path, t)[["gene1", "gene2"]].values.flatten()
            for t in os.listdir(root_path + "/Tissue")
        )
    )
    esm2_genes = set(b.dfs["gene2esm2"].index)
    avail_genes = sorted(all_rna_genes & all_ppi_genes & esm2_genes)
    gene_to_idx = {g: i for i, g in enumerate(avail_genes)}
    b.dfs["gene_to_idx"] = pd.DataFrame(
        list(gene_to_idx.values()), index=list(gene_to_idx.keys()), columns=["idx"]
    )
    return b


@ops.op("LynxKite Graph Analytics", "View other", view="table_view", color="blue")
def view_other(b: core.Bundle):
    b = b.copy()
    b.dfs = {"other": pd.DataFrame(list(b.other.items()), columns=["key", "value"])}
    return b.to_dict()


@pdx_op("Classify mRECIST", slow=True)
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


@pdx_op("Filter to samples with RNA data", slow=True)
def filter_to_samples_with_rna_data(b: core.Bundle) -> core.Bundle:
    b = b.copy()
    valid_pdx = set(b.dfs["rna_df"].columns)
    b.dfs["meta"] = b.dfs["meta"][b.dfs["meta"]["Model"].isin(valid_pdx)].reset_index(drop=True)
    return b


@ops.cached
def load_ppi_for_tissue(root_path: str, tissue: str) -> pd.DataFrame:
    path = os.path.join(root_path, "Tissue", tissue, "edges.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=["gene1", "gene2", "score"])


def gene_interacts_gene_input(
    b: core.Bundle,
    batch_index: int,
    *,
    table_name: core.TableName = "meta",
    tumor_type_column_name: core.ColumnNameByTableName = "Tumor Type",
):
    """
    Args:
        table_name: The table with samples.
        tumor_type_column_name: The column indicating the tissue type for the sample.
    """
    row = b.dfs[table_name].iloc[batch_index]
    ppi_df = load_ppi_for_tissue(b.other["root_path"], row[tumor_type_column_name])
    gene_to_idx = b.dfs["gene_to_idx"]["idx"].to_dict()
    rows, cols = [], []
    for u, v in zip(ppi_df.gene1, ppi_df.gene2):
        if u in gene_to_idx and v in gene_to_idx:
            rows.append(gene_to_idx[u])
            cols.append(gene_to_idx[v])
    edge_idx = torch.tensor([rows, cols], dtype=torch.long)
    edge_idx = pyg_utils.to_undirected(edge_idx)
    edge_idx, _ = pyg_utils.add_self_loops(edge_idx)
    return torch.tensor(edge_idx, dtype=torch.long)


def drug_targets_gene_input(
    b: core.Bundle,
    batch_index: int,
    *,
    table_name: core.TableName = "meta",
    treatment_column_name: core.ColumnNameByTableName = "Treatment",
):
    """
    Args:
        table_name: The table with samples.
        treatment_column_name: The column indicating the treatment drug for the sample.
    """
    gene_to_idx = b.dfs["gene_to_idx"]["idx"].to_dict()
    row = b.dfs[table_name].iloc[batch_index]
    drug = row[treatment_column_name]
    df = b.dfs["drug_gene"]
    dg = df[df["drug_name"] == drug].gene_name
    dg_rows = [gene_to_idx[g] for g in dg if g in gene_to_idx]
    return (
        torch.tensor([[0] * len(dg_rows), dg_rows], dtype=torch.long)
        if dg_rows
        else torch.empty((2, 0), dtype=torch.long)
    )


def disease_assoc_gene_input(
    b: core.Bundle,
    batch_index: int,
    *,
    table_name: core.TableName = "meta",
    tumor_type_column_name: core.ColumnNameByTableName = "Tumor Type",
):
    """
    Args:
        table_name: The table with samples.
        tumor_type_column_name: The column indicating the tissue type for the sample.
    """
    gene_to_idx = b.dfs["gene_to_idx"]["idx"].to_dict()
    row = b.dfs[table_name].iloc[batch_index]
    tumor = row[tumor_type_column_name]
    df = b.dfs["disease"]
    dis = df[df["disease"] == tumor].symbol
    dis_rows = [gene_to_idx[g] for g in dis if g in gene_to_idx]
    return (
        torch.tensor([[0] * len(dis_rows), dis_rows], dtype=torch.long)
        if dis_rows
        else torch.empty((2, 0), dtype=torch.long)
    )


def gene_features_input(
    b: core.Bundle,
    batch_index: int,
    *,
    table_name: core.TableName = "meta",
    pdx_id_column_name: core.ColumnNameByTableName = "Model",
):
    """
    Args:
        table_name: The table with samples.
        pdx_id_column_name: The column indicating the PDX model for the sample.
    """
    row = b.dfs[table_name].iloc[batch_index]
    pdx_id = row[pdx_id_column_name]
    avail_genes = b.dfs["gene_to_idx"].index
    gene2esm2 = b.dfs["gene2esm2"]["embedding"]
    rna_df = b.dfs["rna_df"]
    expr = rna_df[pdx_id].reindex(avail_genes).fillna(0).values
    expr_t = torch.tensor(expr, dtype=torch.float32).view(-1, 1)
    esm2_mat = torch.stack([torch.tensor(gene2esm2[g], dtype=torch.float32) for g in avail_genes])
    return torch.cat([esm2_mat, expr_t], dim=1)


@ops.cached
async def load():
    root_path = "uploads/ode-gnn"
    b = await load_disease_gene_data(root_path=root_path)
    b.merge(await load_drug_gene_data(root_path=root_path))
    b.merge(await load_rna_data(root_path=root_path))
    b.merge(await load_esm2_data(root_path=root_path))
    pct = await load_pct_data(root_path=root_path)
    pct = await classify_mrecist(pct)
    b.merge(pct)
    b = await index_genes(b)
    b = await filter_to_samples_with_rna_data(b)
    b = await classify_mrecist(b)
    return b


def make_graph(b: core.Bundle, batch_index: int):
    """Construct a PyG Data graph for the sample at batch_index in meta."""
    x = gene_features_input(b, batch_index)
    ppi_edges = gene_interacts_gene_input(b, batch_index)
    drug_edges = drug_targets_gene_input(b, batch_index)
    disease_edges = disease_assoc_gene_input(b, batch_index)

    num_genes = x.shape[0]
    feat_dim = x.shape[1]

    # Add a drug node (index num_genes) and a disease node (index num_genes+1)
    x = torch.cat([x, torch.zeros(2, feat_dim)], dim=0)

    # Remap drug edges: source 0 -> drug node index
    if drug_edges.shape[1] > 0:
        drug_edges = drug_edges.clone()
        drug_edges[0] = num_genes
        drug_edges = pyg_utils.to_undirected(drug_edges)

    # Remap disease edges: source 0 -> disease node index
    if disease_edges.shape[1] > 0:
        disease_edges = disease_edges.clone()
        disease_edges[0] = num_genes + 1
        disease_edges = pyg_utils.to_undirected(disease_edges)

    edge_index = torch.cat([ppi_edges, drug_edges, disease_edges], dim=1)
    return Data(x=x, edge_index=edge_index)


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, return_attention=False):
        x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = x.mean(dim=0)  # Global mean pooling
        logit = self.classifier(x).squeeze()
        if return_attention:
            return logit, attn1, attn2
        return logit


def save_model(model, path="gat_model.pt"):
    """Save the trained GAT model to disk."""
    torch.save({"state_dict": model.state_dict(), "in_channels": model.gat1.in_channels}, path)
    print(f"Model saved to {path}")


def load_model(path="gat_model.pt", device=None):
    """Load a saved GAT model from disk."""
    device = device or DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GATModel(in_channels=ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def make_graph_from_strings(b: core.Bundle, drug: str, disease: str):
    """Build a PyG graph for a (drug, disease) pair without needing a meta row."""
    gene_to_idx = b.dfs["gene_to_idx"]["idx"].to_dict()
    avail_genes = b.dfs["gene_to_idx"].index
    gene2esm2 = b.dfs["gene2esm2"]["embedding"]
    num_genes = len(avail_genes)

    # Gene features: ESM2 embeddings + zero expression (no specific PDX model)
    esm2_mat = torch.stack([torch.tensor(gene2esm2[g], dtype=torch.float32) for g in avail_genes])
    x = torch.cat([esm2_mat, torch.zeros(num_genes, 1)], dim=1)
    feat_dim = x.shape[1]

    # PPI edges for the tissue
    ppi_df = load_ppi_for_tissue(b.other["root_path"], disease)
    rows, cols = [], []
    for u, v in zip(ppi_df.gene1, ppi_df.gene2):
        if u in gene_to_idx and v in gene_to_idx:
            rows.append(gene_to_idx[u])
            cols.append(gene_to_idx[v])
    ppi_edges = torch.tensor([rows, cols], dtype=torch.long)
    ppi_edges = pyg_utils.to_undirected(ppi_edges)
    ppi_edges, _ = pyg_utils.add_self_loops(ppi_edges)

    # Drug node (index num_genes)
    df_drug = b.dfs["drug_gene"]
    dg = df_drug[df_drug["drug_name"] == drug].gene_name
    dg_rows = [gene_to_idx[g] for g in dg if g in gene_to_idx]
    drug_edges = (
        torch.tensor([[num_genes] * len(dg_rows), dg_rows], dtype=torch.long)
        if dg_rows
        else torch.empty((2, 0), dtype=torch.long)
    )
    if drug_edges.shape[1] > 0:
        drug_edges = pyg_utils.to_undirected(drug_edges)

    # Disease node (index num_genes + 1)
    df_dis = b.dfs["disease"]
    dis = df_dis[df_dis["disease"] == disease].symbol
    dis_rows = [gene_to_idx[g] for g in dis if g in gene_to_idx]
    disease_edges = (
        torch.tensor([[num_genes + 1] * len(dis_rows), dis_rows], dtype=torch.long)
        if dis_rows
        else torch.empty((2, 0), dtype=torch.long)
    )
    if disease_edges.shape[1] > 0:
        disease_edges = pyg_utils.to_undirected(disease_edges)

    # Add drug + disease nodes
    x = torch.cat([x, torch.zeros(2, feat_dim)], dim=0)
    edge_index = torch.cat([ppi_edges, drug_edges, disease_edges], dim=1)
    return Data(x=x, edge_index=edge_index)


def predict(model, b: core.Bundle, drug: str, disease: str) -> float:
    """Return the predicted probability that the drug is effective for the disease."""
    model.eval()
    graph = make_graph_from_strings(b, drug, disease)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logit = model(graph.x.to(DEVICE), graph.edge_index.to(DEVICE))
    return torch.sigmoid(logit).item()


def predict_with_attribution(model, b: core.Bundle, drug: str, disease: str):
    """Return probability and per-node attention scores.

    Returns:
        prob: Predicted probability that the drug is effective.
        node_scores: DataFrame with gene names (+ drug/disease nodes) and their
                     aggregated attention scores across both GAT layers.
    """
    model.eval()
    graph = make_graph_from_strings(b, drug, disease)
    avail_genes = list(b.dfs["gene_to_idx"].index)
    num_genes = len(avail_genes)
    node_names = avail_genes + [f"drug:{drug}", f"disease:{disease}"]
    num_nodes = len(node_names)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logit, (ei1, aw1), (ei2, aw2) = model(
            graph.x.to(DEVICE), graph.edge_index.to(DEVICE), return_attention=True
        )
    prob = torch.sigmoid(logit).item()

    # Aggregate attention: for each node, sum the attention it receives as a target.
    # aw1/aw2 shape: (num_edges, heads) — average over heads, then scatter-add to target nodes.
    def aggregate_attention(edge_index, attn_weights):
        attn = attn_weights.mean(dim=1) if attn_weights.dim() > 1 else attn_weights
        scores = torch.zeros(num_nodes, device=attn.device)
        targets = edge_index[1]
        scores.scatter_add_(0, targets, attn)
        return scores.cpu()

    scores1 = aggregate_attention(ei1, aw1)
    scores2 = aggregate_attention(ei2, aw2)
    combined = scores1 + scores2

    df = pd.DataFrame({"node": node_names, "attention_score": combined.numpy()})
    df = df.sort_values("attention_score", ascending=False).reset_index(drop=True)
    return prob, df


def top_genes(attribution_df: pd.DataFrame, n: int = 10) -> list[str]:
    """Return the names of the top N most important genes from an attribution DataFrame."""
    genes = attribution_df[~attribution_df["node"].str.contains(":")]
    return genes.head(n)["node"].tolist()


def explain_prediction(drug: str, disease: str, prob: float, gene_list: list[str]) -> str:
    """Ask an OpenAI model to explain why the GNN made its prediction based on top genes."""
    import openai

    effective = "effective" if prob >= 0.5 else "not effective"
    genes_str = ", ".join(gene_list)
    prompt = (
        f"A neural network predicts that {drug} is {effective} against {disease} "
        f"due to genes {genes_str}. "
        f"What is the likely underlying biological explanation?"
    )
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
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
def train(epochs, samples, output):
    """Train the GAT model and save it."""
    import asyncio

    b = asyncio.run(load())
    meta = b.dfs["meta"].head(samples)
    print(meta.head())
    print(f"Using device: {DEVICE}")

    in_channels = gene_features_input(b, 0).shape[1]
    model = GATModel(in_channels=in_channels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for i, rec in enumerate(
            tqdm(meta.itertuples(), total=len(meta), desc=f"Epoch {epoch + 1}")
        ):
            label_str = rec._asdict()["mrecist_class"]
            if label_str not in LABEL_MAP:
                continue
            label = torch.tensor(LABEL_MAP[label_str], device=DEVICE)
            graph = make_graph(b, i)
            with torch.amp.autocast("cuda"):
                out = model(graph.x.to(DEVICE), graph.edge_index.to(DEVICE))
                loss = loss_fn(out, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pred = (out.detach() > 0).float()
            correct += (pred == label).item()
            total += 1

        avg_loss = epoch_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    save_model(model, output)


@cli.command()
@click.argument("drug")
@click.argument("disease")
@click.option("--model-path", default="gat_model.pt", help="Path to the saved model.")
@click.option("--top-n", default=3, help="Number of top genes to include in the explanation.")
def explain(drug, disease, model_path, top_n):
    """Load the model, run inference with attribution, and print an AI explanation."""
    import asyncio

    b = asyncio.run(load())
    model = load_model(model_path)
    prob, attr_df = predict_with_attribution(model, b, drug, disease)
    genes = top_genes(attr_df, n=top_n)

    effective = "effective" if prob >= 0.5 else "not effective"
    print(f"Prediction:\n{drug} is {effective} against {disease} (p={prob:.3f})")
    print(f"Top {top_n} genes: {', '.join(genes)}")

    explanation = explain_prediction(drug, disease, prob, genes)
    print("\nExplanation:")
    print(explanation)


if __name__ == "__main__":
    cli()
