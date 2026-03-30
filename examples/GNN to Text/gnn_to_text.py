"""Custom operations for the "GNN to Text" example."""

import itertools
from lynxkite_core import ops
from lynxkite_graph_analytics import core
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import os.path
import pandas as pd
from tqdm import tqdm

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


def make_graph(b: core.Bundle, drug: str, tumor_type: str):
    # TODO: Based on the *_input functions defined above, construct a PyG graph for the given tumor type.
    # Add node features based on gene_features_input().
    # This graph should include the PPI graph specific to the tissue (as in gene_interacts_gene_input()),
    # and the given drug and disease nodes. (As in disease_assoc_gene_input() and drug_targets_gene_input().)
    return graph


EPOCHS = 3


async def main():
    b = await load()
    meta = b.dfs["meta"]
    print(meta.head())
    # TODO: Define GAT model.
    m = GATModel(...)
    # Create a loss function. We want to do binary classification of the mRECIST class for each (drug, tissue, disease) triplet.
    ...
    for epoch in tqdm(range(EPOCHS)):
        for rec in meta.itertuples():
            tumor_type = rec._asdict()["Tumor Type"]
            drug = rec._asdict()["Treatment"]
            graph = make_graph(b, drug, tumor_type)
            out = m(graph.x, graph.edge_index)
            # Compute loss and optimize.


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
