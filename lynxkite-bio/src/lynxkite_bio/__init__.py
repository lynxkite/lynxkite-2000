"""Graph analytics operations. To be split into separate files when we have more."""

from lynxkite_graph_analytics import Bundle, RelationDefinition
from lynxkite.core import ops
import joblib
import numpy as np
import pandas as pd
import rdkit.Chem
import rdkit.Chem.rdFingerprintGenerator
import rdkit.Chem.Fingerprints.ClusterMols
import scipy

mem = joblib.Memory("../joblib-cache")
ENV = "LynxKite Graph Analytics"
op = ops.op_registration(ENV)


@op("Parse SMILES")
def parse_smiles(bundle: Bundle, *, table="df", smiles_column="SMILES", save_as="mols"):
    """Parse SMILES strings into RDKit molecules."""
    df = bundle.dfs[table]
    mols = [rdkit.Chem.MolFromSmiles(smiles) for smiles in df[smiles_column].dropna()]
    mols = [mol for mol in mols if mol is not None]
    bundle = bundle.copy()
    bundle.dfs[table] = df.assign(**{save_as: mols})
    return bundle


def _get_similarity_matrix(mols):
    mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [(0, mfpgen.GetFingerprint(mol)) for mol in mols]
    similarity_matrix = rdkit.Chem.Fingerprints.ClusterMols.GetDistanceMatrix(
        fps, metric=rdkit.Chem.DataStructs.TanimotoSimilarity, isSimilarity=1
    )
    return scipy.spatial.distance.squareform(similarity_matrix)


@op("Graph from molecule similarity")
def graph_from_similarity(
    bundle: Bundle, *, table="df", mols_column="mols", average_degree=10
):
    df = bundle.dfs[table]
    mols = df[mols_column]
    similarity_matrix = _get_similarity_matrix(mols)
    i_idx, j_idx = np.triu_indices_from(similarity_matrix, k=1)
    sim_values = similarity_matrix[i_idx, j_idx]
    N = int(average_degree * len(mols))
    top_n_idx = np.argsort(sim_values)[-N:]
    top_n_pairs = [(i_idx[k], j_idx[k], sim_values[k]) for k in top_n_idx]
    edges = pd.DataFrame(top_n_pairs, columns=["source", "target", "similarity"])
    nodes = df.copy()
    nodes.index.name = "id"
    bundle = Bundle(
        dfs={"edges": edges, "nodes": nodes},
        relations=[
            RelationDefinition(
                df="edges",
                source_column="source",
                target_column="target",
                source_table="nodes",
                target_table="nodes",
                source_key="id",
                target_key="id",
            )
        ],
    )
    return bundle
