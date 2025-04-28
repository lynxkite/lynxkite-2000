import os
import pickle
from lynxkite.core.ops import op
from matplotlib import pyplot as plt
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen, Lipinski
from rdkit import DataStructs
import math
import io
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


@op("LynxKite Graph Analytics", "View mol filter", view="matplotlib", slow=True)
def mol_filter(
    bundle,
    *,
    table_name: str,
    SMILES_Column: str,
    mols_per_row: int,
    filter_smarts: str = None,
    filter_smiles: str = None,
    highlight: bool = True,
):
    """
    Draws a grid of molecules in square boxes, with optional filtering and substructure highlighting.

    Parameters:
    - bundle: data bundle containing a DataFrame in bundle.dfs[table_name]
    - table_name: name of the table in bundle.dfs
    - column_name: column containing SMILES strings
    - mols_per_row: number of molecules per row in the grid
    - filter_smarts: SMARTS pattern to filter and highlight
    - filter_smiles: SMILES substructure to filter and highlight (if filter_smarts is None)
    - highlight: whether to highlight matching substructures
    """
    # get DataFrame
    df = bundle.dfs[table_name].copy()
    df["mol"] = df[SMILES_Column].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].reset_index(drop=True)

    # compile substructure query if provided
    query = None
    if filter_smarts:
        query = Chem.MolFromSmarts(filter_smarts)
    elif filter_smiles:
        query = Chem.MolFromSmiles(filter_smiles)

    # compute properties and legends
    df["MW"] = df["mol"].apply(Descriptors.MolWt)
    df["logP"] = df["mol"].apply(Crippen.MolLogP)
    df["HBD"] = df["mol"].apply(Lipinski.NumHDonors)
    df["HBA"] = df["mol"].apply(Lipinski.NumHAcceptors)

    legends = []
    for _, row in df.iterrows():
        mol = row["mol"]
        # filter by substructure
        if query and not mol.HasSubstructMatch(query):
            continue

        # find atom and bond matches
        atom_ids, bond_ids = [], []
        if highlight and query:
            atom_ids = list(mol.GetSubstructMatch(query))
            # find bonds where both ends are in atom_ids
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in atom_ids and a2 in atom_ids:
                    bond_ids.append(bond.GetIdx())

        legend = (
            f"{row['Name']}  pIC50={row['pIC50']:.2f}\n"
            f"MW={row['MW']:.1f}, logP={row['logP']:.2f}\n"
            f"HBD={row['HBD']}, HBA={row['HBA']}"
        )
        legends.append((mol, legend, atom_ids, bond_ids))

    if not legends:
        raise ValueError("No molecules passed the filter.")

    # draw each filtered molecule
    images = []
    for mol, legend, atom_ids, bond_ids in legends:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 350)
        opts = drawer.drawOptions()
        opts.legendFontSize = 200
        drawer.DrawMolecule(mol, legend=legend, highlightAtoms=atom_ids, highlightBonds=bond_ids)
        drawer.FinishDrawing()

        sub_png = drawer.GetDrawingText()
        sub_img = Image.open(io.BytesIO(sub_png))
        images.append(sub_img)

    plot_gallery(images, num_cols=mols_per_row)


@op("LynxKite Graph Analytics", "Lipinski filter")
def lipinski_filter(bundle, *, table_name: str, column_name: str, strict_lipinski: bool = True):
    # copy bundle and get DataFrame
    bundle = bundle.copy()
    df = bundle.dfs[table_name].copy()
    df["mol"] = df[column_name].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].reset_index(drop=True)

    # compute properties
    df["MW"] = df["mol"].apply(Descriptors.MolWt)
    df["logP"] = df["mol"].apply(Crippen.MolLogP)
    df["HBD"] = df["mol"].apply(Lipinski.NumHDonors)
    df["HBA"] = df["mol"].apply(Lipinski.NumHAcceptors)

    # compute a boolean pass/fail for Lipinski
    df["pass_lipinski"] = (
        (df["MW"] <= 500) & (df["logP"] <= 5) & (df["HBD"] <= 5) & (df["HBA"] <= 10)
    )
    df = df.drop("mol", axis=1)

    # if strict_lipinski, drop those that fail
    if strict_lipinski:
        failed = df.loc[~df["pass_lipinski"], column_name].tolist()
        df = df[df["pass_lipinski"]].reset_index(drop=True)
        if failed:
            print(f"Dropped {len(failed)} molecules that failed Lipinski: {failed}")

    return df


@op("LynxKite Graph Analytics", "View mol image", view="matplotlib", slow=True)
def mol_image(bundle, *, table_name: str, smiles_column: str, mols_per_row: int):
    df = bundle.dfs[table_name].copy()
    df["mol"] = df[smiles_column].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].reset_index(drop=True)
    df["MW"] = df["mol"].apply(Descriptors.MolWt)
    df["logP"] = df["mol"].apply(Crippen.MolLogP)
    df["HBD"] = df["mol"].apply(Lipinski.NumHDonors)
    df["HBA"] = df["mol"].apply(Lipinski.NumHAcceptors)

    legends = []
    for _, row in df.iterrows():
        legends.append(
            f"{row['Name']}  pIC50={row['pIC50']:.2f}\n"
            f"MW={row['MW']:.1f}, logP={row['logP']:.2f}\n"
            f"HBD={row['HBD']}, HBA={row['HBA']}"
        )

    mols = df["mol"].tolist()
    if not mols:
        raise ValueError("No valid molecules to draw.")

    # --- draw each molecule into its own sub‐image and paste ---
    images = []
    for mol, legend in zip(mols, legends):
        # draw one molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 350)
        opts = drawer.drawOptions()
        opts.legendFontSize = 200
        drawer.DrawMolecule(mol, legend=legend)
        drawer.FinishDrawing()
        sub_png = drawer.GetDrawingText()
        sub_img = Image.open(io.BytesIO(sub_png))
        images.append(sub_img)

    plot_gallery(images, num_cols=mols_per_row)


def plot_gallery(images, num_cols):
    num_rows = math.ceil(len(images) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()


@op("LynxKite Graph Analytics", "Train QSAR model")
def build_qsar_model(
    bundle,
    *,
    table_name: str,
    smiles_col: str,
    target_col: str,
    fp_type: str,
    radius: int = 2,
    n_bits: int = 2048,
    test_size: float = 0.2,
    random_state: int = 42,
    out_dir: str = "Models",
):
    """
    Train and save a RandomForest QSAR model using one fingerprint type.

    Parameters
    ----------
    bundle : any
        An object with a dict‐like attribute `.dfs` mapping table names to DataFrames.
    table_name : str
        Key into bundle.dfs to get the DataFrame.
    smiles_col : str
        Name of the column containing SMILES strings.
    target_col : str
        Name of the column containing the numeric response.
    fp_type : str
        Fingerprint to compute: "ecfp", "rdkit", "torsion", "atompair", or "maccs".
    radius : int
        Radius for the Morgan (ECFP) fingerprint.
    n_bits : int
        Bit‐vector length for all fp types except MACCS (167).
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Random seed for reproducibility.
    out_dir : str
        Directory in which to save `qsar_model_<fp_type>.pkl`.

    Returns
    -------
    model : RandomForestRegressor
        The trained QSAR model.
    metrics_df : pandas.DataFrame
        R², MAE and RMSE on train and test splits.
    """
    # 1) load and sanitize data
    df = bundle.dfs.get(table_name)
    if df is None:
        raise KeyError(f"Table '{table_name}' not found in bundle.dfs")
    df = df.copy()
    df["mol"] = df[smiles_col].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid molecules in '{smiles_col}'")

    # 2) create a fixed train/test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    # 3) featurize
    fps = []
    for mol in df["mol"]:
        if fp_type == "ecfp":
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        elif fp_type == "rdkit":
            bv = Chem.RDKFingerprint(mol, fpSize=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        elif fp_type == "torsion":
            bv = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        elif fp_type == "atompair":
            bv = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        elif fp_type == "maccs":
            bv = Chem.MACCSkeys.GenMACCSKeys(mol)  # 167 bits
            arr = np.zeros((167,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        else:
            raise ValueError(f"Unsupported fingerprint type: '{fp_type}'")
        fps.append(arr)

    X = np.vstack(fps)
    y = df[target_col].values

    # 4) split features/labels
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 5) train RandomForest
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    # 6) compute performance metrics
    def _metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mse),
        }

    train_m = _metrics(y_train, model.predict(X_train))
    test_m = _metrics(y_test, model.predict(X_test))
    metrics_df = pd.DataFrame([{"split": "train", **train_m}, {"split": "test", **test_m}])

    # 7) save the model
    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, f"qsar_model_{fp_type}.pkl")
    with open(model_file, "wb") as fout:
        pickle.dump(model, fout)

    print(f"Trained & saved QSAR model for '{fp_type}' → {model_file}")
    return metrics_df
