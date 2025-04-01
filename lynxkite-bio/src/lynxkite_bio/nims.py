"""Wrappers for BioNeMo NIMs."""

from enum import Enum
from lynxkite_graph_analytics import Bundle
from lynxkite.core import ops
import joblib
import os

NIM_URLS = os.environ.get("NIM_URLS", "http://localhost:8000").split(",")

mem = joblib.Memory(".joblib-cache")
ENV = "LynxKite Graph Analytics"
op = ops.op_registration(ENV)


class MSASearchTypes(Enum):
    ALPHAFOLD2 = "ALPHAFOLD2"
    ESM2 = "ESM2"


class AlignmentFormats(Enum):
    FASTA = "fasta"
    A3M = "a3m"
    STOCKHOLM = "stockholm"
    CLUSTAL = "clustal"
    PDB = "pdb"
    PIR = "pir"
    MSF = "msf"
    TSV = "tsv"


@op("MSA-search")
@mem.cache
def msa_search(
    bundle: Bundle,
    *,
    protein_table: str,
    protein_column: str,
    e_value: float = 0.0001,
    iterations: int = 1,
    search_type: MSASearchTypes = MSASearchTypes.ALPHAFOLD2,
    output_alignment_formats: list[AlignmentFormats] = [
        AlignmentFormats.FASTA,
        AlignmentFormats.A3M,
    ],
    databases: str = '["Uniref30_2302", "colabfold_envdb_202108", "PDB70_220313"]',
):
    bundle = bundle.copy()
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
    use_templates: bool = False,
    relaxed_prediction: bool = False,
    databases: str = '["Uniref30_2302", "colabfold_envdb_202108", "PDB70_220313"]',
):
    bundle = bundle.copy()
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
    num_poses=10,
    time_divisions=20,
    num_steps=18,
):
    return proteins
