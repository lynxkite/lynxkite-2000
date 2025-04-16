"""Wrappers for BioNeMo NIMs."""

from lynxkite_graph_analytics import Bundle
from lynxkite.core import ops
import joblib
import httpx
import pandas as pd
import os


mem = joblib.Memory(".joblib-cache")
ENV = "LynxKite Graph Analytics"
op = ops.op_registration(ENV)

key = os.getenv("NVCF_RUN_KEY")


async def query_bionemo_nim(
    url: str,
    payload: dict,
):
    headers = {
        "Authorization": f"Bearer {key}",
        "NVCF-POLL-SECONDS": "500",
        "Content-Type": "application/json",
    }
    try:
        print(f"Sending request to {url}")
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(url, json=payload, headers=headers)
        print(f"Received response from {url}", response.status_code)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise ValueError(f"Query failed: {e}")


@op("MSA-search")
@mem.cache
async def msa_search(
    bundle: Bundle,
    *,
    protein_table: str,
    protein_column: str,
    e_value: float = 0.0001,
    iterations: int = 1,
    search_type: str = "alphafold2",
    output_alignment_formats: str = "a3m",
    databases: str = "Uniref30_2302,colabfold_envdb_202108",
):
    bundle = bundle.copy()
    bundle.dfs[protein_table]["alignments"] = None

    formats = [format.strip() for format in output_alignment_formats.split(",")]
    dbs = [db.strip() for db in databases.split(",")]

    for idx, protein_sequence in enumerate(bundle.dfs[protein_table][protein_column]):
        print(f"Processing protein {idx + 1}/{len(bundle.dfs[protein_table])}")

        response = await query_bionemo_nim(
            url="https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict",
            payload={
                "sequence": protein_sequence,
                "e_value": e_value,
                "iterations": iterations,
                "search_type": search_type,
                "output_alignment_formats": formats,
                "databases": dbs,
            },
        )

        bundle.dfs[protein_table].at[idx, "alignments"] = response["alignments"]

    return bundle


@op("Query OpenFold2")
@mem.cache
async def query_openfold2(
    bundle: Bundle,
    *,
    protein_table: str,
    protein_column: str,
    alignment_table: str,
    alignment_column: str,
    selected_models: str = "1,2",
    relax_prediction: bool = False,
):
    bundle = bundle.copy()

    bundle.dfs[protein_table]["folded_protein"] = None
    selected_models_list = [int(model) for model in selected_models.split(",")]

    for idx in range(len(bundle.dfs[protein_table])):
        print(f"Processing protein {idx + 1}/{len(bundle.dfs[protein_table])}")

        protein = bundle.dfs[protein_table][protein_column].iloc[idx]
        alignments = bundle.dfs[alignment_table][alignment_column].iloc[idx]

        response = await query_bionemo_nim(
            url="https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template",
            payload={
                "sequence": protein,
                "alignments": alignments,
                "selected_models": selected_models_list,
                "relax_prediction": relax_prediction,
            },
        )

        folded_protein = response["structures_in_ranked_order"].pop(0)["structure"]
        bundle.dfs[protein_table].at[idx, "folded_protein"] = folded_protein

    bundle.dfs["openfold"] = pd.DataFrame()

    return bundle


@op("View molecule", view="molecule")
def view_molecule(
    bundle: Bundle,
    *,
    molecule_table: str,
    molecule_column: str,
    row_index: int = 0,
):
    molecule_data = bundle.dfs[molecule_table][molecule_column].iloc[row_index]

    return {
        "data": molecule_data,
        "format": "pdb"
        if molecule_data.startswith("ATOM")
        else "sdf"
        if molecule_data.startswith("CTfile")
        else "smiles",
    }


@op("Query GenMol")
@mem.cache
async def query_genmol(
    bundle: Bundle,
    *,
    molecule_table: str,
    molecule_column: str,
    num_molecules: int = 5,
    temperature: float = 1.0,
    noise: float = 0.2,
    step_size: int = 4,
    scoring: str = "QED",
):
    bundle = bundle.copy()

    response = await query_bionemo_nim(
        url="https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate",
        payload={
            "smiles": bundle.dfs[molecule_table][molecule_column].iloc[0],
            "num_molecules": num_molecules,
            "temperature": temperature,
            "noise": noise,
            "step_size": step_size,
            "scoring": scoring,
        },
    )
    generated_ligands = "\n".join([v["smiles"] for v in response["molecules"]])
    bundle.dfs[molecule_table]["ligands"] = generated_ligands
    return bundle


@op("Query DiffDock")
@mem.cache
async def query_diffdock(
    proteins: Bundle,
    ligands: Bundle,
    *,
    protein_table: str,
    protein_column: str,
    ligand_table: str,
    ligand_column: str,
    ligand_file_type: str = "txt",
    num_poses=10,
    time_divisions=20,
    num_steps=18,
):
    response = await query_bionemo_nim(
        url="https://health.api.nvidia.com/v1/biology/mit/diffdock",
        payload={
            "protein": proteins.dfs[protein_table][protein_column].iloc[0],
            "ligand": ligands.dfs[ligand_table][ligand_column].iloc[0],
            "ligand_file_type": ligand_file_type,
            "num_poses": num_poses,
            "time_divisions": time_divisions,
            "num_steps": num_steps,
        },
    )
    bundle = Bundle()
    bundle.dfs["diffdock_table"] = pd.DataFrame()
    bundle.dfs["diffdock_table"]["protein"] = [response["protein"]] * len(response["status"])
    bundle.dfs["diffdock_table"]["ligand"] = [response["ligand"]] * len(response["status"])
    bundle.dfs["diffdock_table"]["trajectory"] = response["trajectory"]
    bundle.dfs["diffdock_table"]["ligand_positions"] = response["ligand_positions"]
    bundle.dfs["diffdock_table"]["position_confidence"] = response["position_confidence"]
    bundle.dfs["diffdock_table"]["status"] = response["status"]

    return bundle
