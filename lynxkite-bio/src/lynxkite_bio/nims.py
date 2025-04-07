"""Wrappers for BioNeMo NIMs."""

from lynxkite_graph_analytics import Bundle
from lynxkite.core import ops
import joblib
import requests
import pandas as pd
import os


mem = joblib.Memory(".joblib-cache")
ENV = "LynxKite Graph Analytics"
op = ops.op_registration(ENV)

key = os.getenv("NVCF_RUN_KEY")


def query_bionemo_nim(
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
        response = requests.post(url, json=payload, headers=headers)
        print(f"Received response from {url}", response.status_code)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Query failed: {e}")


@op("MSA-search")
@mem.cache
def msa_search(
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
    response = query_bionemo_nim(
        url="https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict",
        payload={
            "sequence": bundle.dfs[protein_table][protein_column].iloc[0],
            "e_value": e_value,
            "iterations": iterations,
            "search_type": search_type,
            "output_alignment_formats": [
                format for format in output_alignment_formats.split(",")
            ],
            "databases": [db for db in databases.split(",")],
        },
    )
    bundle.dfs[protein_table]["alignments"] = [response["alignments"]]
    return bundle


@op("Query OpenFold2")
@mem.cache
def query_openfold2(
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
    protein = bundle.dfs[protein_table][protein_column].iloc[0]
    alignments = bundle.dfs[alignment_table][alignment_column].iloc[0]
    selected_models = [int(model) for model in selected_models.split(",")]
    response = query_bionemo_nim(
        url="https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template",
        payload={
            "sequence": protein,
            "alignments": alignments,
            "selected_models": selected_models,
            "relax_prediction": relax_prediction,
        },
    )
    folded_protein = response["structures_in_ranked_order"].pop(0)["structure"]
    bundle.dfs[protein_table]["folded_protein"] = folded_protein
    return bundle


@op("View molecules", view="visualization")
def view_molecules(
    bundle: Bundle,
    *,
    molecule_table: str,
    molecule_column: str,
    color="spectrum",
):
    return {
        "series": [
            {
                "type": "pie",
                "radius": ["40%", "70%"],
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "data": [
                    {"value": 2, "name": "Hydrogen"},
                    {"value": 1, "name": "Sulfur"},
                    {"value": 4, "name": "Oxygen"},
                ],
            }
        ]
    }


@op("Known drug")
def known_drug(*, drug_name: str):
    return Bundle()


@op("Query GenMol")
@mem.cache
def query_genmol(
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

    response = query_bionemo_nim(
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
def query_diffdock(
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
    response = query_bionemo_nim(
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
    bundle.dfs["diffdock_table"]["protein"] = [response["protein"]] * len(
        response["status"]
    )
    bundle.dfs["diffdock_table"]["ligand"] = [response["ligand"]] * len(
        response["status"]
    )
    bundle.dfs["diffdock_table"]["trajectory"] = response["trajectory"]
    bundle.dfs["diffdock_table"]["ligand_positions"] = response["ligand_positions"]
    bundle.dfs["diffdock_table"]["position_confidence"] = response[
        "position_confidence"
    ]
    bundle.dfs["diffdock_table"]["status"] = response["status"]

    return bundle
