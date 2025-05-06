import base64
import io
import sys
from io import StringIO
from operator import itemgetter
from typing import List
from typing import Tuple
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Cluster import Butina
from rdkit.rdBase import BlockLogs

import pandas as pd
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose


def smi2mol_with_errors(smi: str) -> Tuple[Mol, str]:
    """Parse SMILES and return any associated errors or warnings

    :param smi: input SMILES
    :return: tuple of RDKit molecule, warning or error
    """
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromSmiles(smi)
    err = sio.getvalue()
    sio = sys.stderr = StringIO()
    sys.stderr = sys.__stderr__
    return mol, err


def count_fragments(mol: Mol) -> int:
    """Count the number of fragments in a molecule

    :param mol: RDKit molecule
    :return: number of fragments
    """
    return len(Chem.GetMolFrags(mol, asMols=True))


def get_largest_fragment(mol: Mol) -> Mol:
    """Return the fragment with the largest number of atoms

    :param mol: RDKit molecule
    :return: RDKit molecule with the largest number of atoms
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_mw_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_mw_list.sort(key=itemgetter(0), reverse=True)
    return frag_mw_list[0][1]


# ----------- Clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
def taylor_butina_clustering(
    fp_list: List[DataStructs.ExplicitBitVect], cutoff: float = 0.65
) -> List[int]:
    """Cluster a set of fingerprints using the RDKit Taylor-Butina implementation

    :param fp_list: a list of fingerprints
    :param cutoff: distance cutoff (1 - Tanimoto similarity)
    :return: a list of cluster ids
    """
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    cluster_res = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = np.zeros(nfps, dtype=int)
    for cluster_num, cluster in enumerate(cluster_res):
        for member in cluster:
            cluster_id_list[member] = cluster_num
    return cluster_id_list.tolist()


# ----------- Atom tagging
def label_atoms(mol: Mol, labels: List[str]) -> Mol:
    """Label atoms when depicting a molecule

    :param mol: input molecule
    :param labels: labels, one for each atom
    :return: molecule with labels
    """
    [atm.SetProp("atomNote", "") for atm in mol.GetAtoms()]
    for atm in mol.GetAtoms():
        idx = atm.GetIdx()
        mol.GetAtomWithIdx(idx).SetProp("atomNote", f"{labels[idx]}")
    return mol


def tag_atoms(mol: Mol, atoms_to_tag: List[int], tag: str = "x") -> Mol:
    """Tag atoms with a specified string

    :param mol: input molecule
    :param atoms_to_tag: indices of atoms to tag
    :param tag: string to use for the tags
    :return: molecule with atoms tagged
    """
    [atm.SetProp("atomNote", "") for atm in mol.GetAtoms()]
    [mol.GetAtomWithIdx(idx).SetProp("atomNote", tag) for idx in atoms_to_tag]
    return mol


# ----------- Logging
def rd_shut_the_hell_up() -> None:
    """Make the RDKit be a bit more quiet

    :return: None
    """
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


def demo_block_logs() -> None:
    """An example of another way to turn off RDKit logging

    :return: None
    """
    block = BlockLogs()
    # do stuff
    del block


# ----------- Image generation
def boxplot_base64_image(dist: np.ndarray, x_lim: list[int] = [0, 10]) -> str:
    """
    Plot a distribution as a seaborn boxplot and save the resulting image as a base64 image.

    Parameters:
    dist (np.ndarray): The distribution data to plot.
    x_lim (list[int]): The x-axis limits for the boxplot.

    Returns:
    str: The base64 encoded image string.
    """
    sns.set(rc={"figure.figsize": (3, 1)})
    sns.set_style("whitegrid")
    ax = sns.boxplot(x=dist)
    ax.set_xlim(x_lim[0], x_lim[1])
    s = io.BytesIO()
    plt.savefig(s, format="png", bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s


def mol_to_base64_image(mol: Chem.Mol) -> str:
    """
    Convert an RDKit molecule to a base64 encoded image string.

    Parameters:
    mol (Chem.Mol): The RDKit molecule to convert.

    Returns:
    str: The base64 encoded image string.
    """
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    text = drawer.GetDrawingText()
    im_text64 = base64.b64encode(text).decode("utf8")
    img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
    return img_str


def cleanup_fragment(mol: Mol) -> Tuple[Mol, int]:
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count


def generate_fragments(mol: Mol) -> pd.DataFrame:
    """
    Generate fragments using the RDKit
    :param mol: RDKit molecule
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    # Generate molecule fragments
    frag_list = FragmentMol(mol)
    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    # The output of Fragment mol is contained in single molecules.  Extract the largest fragment from each molecule
    flat_frag_list = [get_largest_fragment(x) for x in flat_frag_list]
    # Keep fragments where the number of atoms in the fragment is at least 2/3 of the number fragments in
    # input molecule
    num_mol_atoms = mol.GetNumAtoms()
    flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > 0.67]
    # remove atom map numbers from the fragments
    flat_frag_list = [cleanup_fragment(x) for x in flat_frag_list]
    # Convert fragments to SMILES
    frag_smiles_list = [[Chem.MolToSmiles(x), x.GetNumAtoms(), y] for (x, y) in flat_frag_list]
    # Add the input molecule to the fragment list
    frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    # Put the results into a Pandas dataframe
    frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroupgs"])
    # Remove duplicate fragments
    frag_df = frag_df.drop_duplicates("Scaffold")
    return frag_df


def find_scaffolds(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    # Loop over molecules and generate fragments, fragments for each molecule are returned as a Pandas dataframe
    df_list = []
    for smiles, name, mol in df_in[["SMILES", "Name", "mol"]].values:
        tmp_df = generate_fragments(mol).copy()
        tmp_df["Name"] = name
        tmp_df["SMILES"] = smiles
        df_list.append(tmp_df)
    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list)
    # Collect scaffolds
    scaffold_list = []
    for k, v in mol_df.groupby("Scaffold"):
        scaffold_list.append([k, len(v.Name.unique()), v.NumAtoms.values[0]])
    scaffold_df = pd.DataFrame(scaffold_list, columns=["Scaffold", "Count", "NumAtoms"])
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    num_df_rows = len(df_in)  # noqa: F841
    scaffold_df = scaffold_df.query(f"Count <= {num_df_rows}")
    # Sort scaffolds by frequency
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    return mol_df, scaffold_df


def get_molecules_with_scaffold(
    scaffold: str, mol_df: pd.DataFrame, activity_df: pd.DataFrame
) -> Tuple[List[str], pd.DataFrame]:
    """
    Associate molecules with scaffolds
    :param scaffold: scaffold SMILES
    :param mol_df: dataframe with molecules and scaffolds, returned by find_scaffolds()
    :param activity_df: dataframe with [SMILES, Name, pIC50] columns
    :return: list of core(s) with R-groups labeled, dataframe with [SMILES, Name, pIC50]
    """
    match_df = mol_df.query("Scaffold == @scaffold")
    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    rgroup_match, rgroup_miss = RGroupDecompose(scaffold_mol, merge_df.mol, asSmiles=True)
    if len(rgroup_match):
        rgroup_df = pd.DataFrame(rgroup_match)
        return rgroup_df.Core.unique(), merge_df[["SMILES", "Name", "pIC50"]]
    else:
        return [], merge_df[["SMILES", "Name", "pIC50"]]
