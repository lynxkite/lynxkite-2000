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
from rdkit.Chem import MACCSkeys


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


def predict_with_ci(model, X, confidence=0.95):
    """
    Calculates predictions and confidence intervals for a RandomForestRegressor.
    (Implementation is the same as in the previous answer)
    """
    # Get predictions from each individual tree
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # Calculate mean prediction
    y_pred_mean = np.mean(tree_preds, axis=0)
    # Calculate percentiles for confidence interval
    alpha = (1.0 - confidence) / 2.0
    lower_percentile = alpha * 100
    upper_percentile = (1.0 - alpha) * 100
    y_pred_lower = np.percentile(tree_preds, lower_percentile, axis=0)
    y_pred_upper = np.percentile(tree_preds, upper_percentile, axis=0)
    return y_pred_mean, y_pred_lower, y_pred_upper


# --- End of predict_with_ci definition ---


@op("LynxKite Graph Analytics", "Train QSAR2")
def build_qsar_model2(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    target_col: str,
    fp_type: str,
    radius: int = 2,
    n_bits: int = 2048,
    test_size: float = 0.2,
    random_state: int = 42,
    out_dir: str = "Models",
    confidence: float = 0.95,
):
    """
    Train/save RandomForest QSAR model, returning the model and a results DataFrame.

    The results DataFrame contains per-point data ('actual', 'predicted',
    'lower_ci', 'upper_ci', 'split') AND repeated summary metrics for each
    split ('split_R2', 'split_MAE', 'split_RMSE').

    Parameters
    ----------
    (Parameters are the same as before)
    bundle : any
    table_name : str
    smiles_col : str
    target_col : str
    fp_type : str
    radius : int
    n_bits : int
    test_size : float
    random_state : int
    out_dir : str
    confidence : float, optional

    Returns
    -------
    model : RandomForestRegressor
        The trained QSAR model.
    results_df : pandas.DataFrame
        DataFrame containing columns: 'actual', 'predicted', 'lower_ci',
        'upper_ci', 'split', 'split_R2', 'split_MAE', 'split_RMSE'.
        The metric columns repeat the overall metric for the corresponding split.
    """
    # Steps 1-5: Load data, split, featurize, split features, train model
    # (Code is identical to previous versions up to model training)
    # ... (load data, sanitize, split indices) ...
    # df = bundle.dfs.get(table_name)
    df = df.copy()
    if df is None:
        raise KeyError("Table not found")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df.dropna(subset=[target_col, smiles_col], inplace=True)
    df["mol"] = df[smiles_col].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid molecules or targets")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    print(f"Featurizing using {fp_type}...")
    fps = []
    valid_indices = []
    for i, mol in enumerate(df["mol"]):
        try:
            # ... (fp generation logic as before) ...
            if fp_type == "ecfp":
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                current_n_bits = n_bits
            elif fp_type == "rdkit":
                bv = Chem.RDKFingerprint(mol, fpSize=n_bits)
                current_n_bits = n_bits
            elif fp_type == "torsion":
                bv = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
                current_n_bits = n_bits
            elif fp_type == "atompair":
                bv = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
                current_n_bits = n_bits
            elif fp_type == "maccs":
                bv = MACCSkeys.GenMACCSKeys(mol)  # 167 bits
                current_n_bits = 167
            else:
                raise ValueError(f"Unsupported fp type: '{fp_type}'")

            arr = np.zeros((current_n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            fps.append(arr)
            valid_indices.append(i)
        except Exception as e:
            print(f"Warning: Featurization failed index {i}. Skipping. Error: {e}")
            continue
    if not fps:
        raise ValueError("No molecules featurized.")
    X = np.vstack(fps)
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    y = df_filtered[target_col].values

    # original_indices_set = set(valid_indices)

    train_idx_filtered = [
        i for i, original_idx in enumerate(valid_indices) if original_idx in train_idx
    ]
    test_idx_filtered = [
        i for i, original_idx in enumerate(valid_indices) if original_idx in test_idx
    ]

    X_train, y_train = X[train_idx_filtered], y[train_idx_filtered]
    X_test, y_test = X[test_idx_filtered], y[test_idx_filtered]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Train or test split empty after filtering.")

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    # 6) Compute predictions and *summary* performance metrics
    print("Calculating predictions and metrics...")
    y_pred_train, lower_ci_train, upper_ci_train = predict_with_ci(model, X_train, confidence)
    y_pred_test, lower_ci_test, upper_ci_test = predict_with_ci(model, X_test, confidence)

    def _metrics(y_true, y_pred_mean):
        # (Same helper function as before)
        y_true = np.ravel(y_true)
        y_pred_mean = np.ravel(y_pred_mean)
        if len(y_true) == 0:
            return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
        mse = mean_squared_error(y_true, y_pred_mean)
        return {
            "R2": r2_score(y_true, y_pred_mean),
            "MAE": mean_absolute_error(y_true, y_pred_mean),
            "RMSE": np.sqrt(mse),
        }

    train_metrics_dict = _metrics(y_train, y_pred_train)
    test_metrics_dict = _metrics(y_test, y_pred_test)

    # 7) Create results DataFrames and ADD metrics columns
    train_results = pd.DataFrame(
        {
            "actual": y_train,
            "predicted": y_pred_train,
            "lower_ci": lower_ci_train,
            "upper_ci": upper_ci_train,
            "split": "train",
        }
    )
    # Add repeated metrics
    for metric, value in train_metrics_dict.items():
        train_results[f"split_{metric}"] = value

    test_results = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred_test,
            "lower_ci": lower_ci_test,
            "upper_ci": upper_ci_test,
            "split": "test",
        }
    )
    # Add repeated metrics
    for metric, value in test_metrics_dict.items():
        test_results[f"split_{metric}"] = value

    # Concatenate into the final DataFrame
    results_df = pd.concat([train_results, test_results], ignore_index=True)

    # 8) Save the model (same as before)
    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, f"qsar_model_{fp_type}.pkl")
    try:
        with open(model_file, "wb") as fout:
            pickle.dump(model, fout)
        print(f"Trained & saved QSAR model for '{fp_type}' -> {model_file}")
    except Exception as e:
        print(f"Error saving model to {model_file}: {e}")

    return results_df


@op("LynxKite Graph Analytics", "plot qsar", view="matplotlib")
def plot_qsar(results_df: pd.DataFrame):
    """
    Plots actual vs. predicted values from a QSAR results DataFrame.

    Requires a single positional argument: the results DataFrame. All other
    parameters are optional keyword arguments. It extracts summary metrics
    directly from columns ('split_R2', 'split_MAE', 'split_RMSE')
    expected within the results_df.
    """
    title = "QSAR Model Performance: Actual vs. Predicted"
    xlabel = "Actual Values"
    ylabel = "Predicted Values"
    show_metrics = True

    if not isinstance(results_df, pd.DataFrame):
        raise TypeError(
            "plot_qsar() missing 1 required positional argument: 'results_df' or the provided argument is not a pandas DataFrame."
        )

    required_cols = ["actual", "predicted", "lower_ci", "upper_ci", "split"]
    if not all(col in results_df.columns for col in required_cols):
        raise ValueError(f"Invalid 'results_df'. Must contain columns: {required_cols}")

    metric_cols = ["split_R2", "split_MAE", "split_RMSE"]
    metrics_available = all(col in results_df.columns for col in metric_cols)
    if show_metrics and not metrics_available:
        print(
            f"Warning: Metrics display requested, but one or more metric columns ({metric_cols}) are missing in results_df."
        )

    # --- Prepare Data ---
    train_data = results_df[results_df["split"] == "train"]
    test_data = results_df[results_df["split"] == "test"]
    can_plot_train = not train_data.empty
    can_plot_test = not test_data.empty

    if not can_plot_train and not can_plot_test:
        print("Warning: Both training and test data subsets are empty. Cannot generate plot.")
        return  # Exit function early if no data

    # --- Create Plot (Internal Figure/Axes) ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Plotting Logic ---
    # (Draws scatter, error bars, line, grid, labels, title, legend on 'ax')
    if can_plot_train:
        train_error = [
            train_data["predicted"] - train_data["lower_ci"],
            train_data["upper_ci"] - train_data["predicted"],
        ]
        ax.scatter(
            train_data["actual"],
            train_data["predicted"],
            label="Train",
            alpha=0.6,
            s=30,
            edgecolors="w",
            linewidth=0.5,
        )
        ax.errorbar(
            train_data["actual"],
            train_data["predicted"],
            yerr=train_error,
            fmt="none",
            ecolor="tab:blue",
            label="_nolegend_",
            capsize=0,
            elinewidth=1,
        )

    if can_plot_test:
        test_error = [
            test_data["predicted"] - test_data["lower_ci"],
            test_data["upper_ci"] - test_data["predicted"],
        ]
        ax.scatter(
            test_data["actual"],
            test_data["predicted"],
            label="Test",
            alpha=0.8,
            s=40,
            edgecolors="w",
            linewidth=0.5,
        )
        ax.errorbar(
            test_data["actual"],
            test_data["predicted"],
            yerr=test_error,
            fmt="none",
            ecolor="tab:orange",
            label="_nolegend_",
            capsize=0,
            elinewidth=1,
        )

    all_actual = results_df["actual"].dropna()
    all_pred_ci = pd.concat(
        [results_df["predicted"], results_df["lower_ci"], results_df["upper_ci"]]
    ).dropna()
    all_values = pd.concat([all_actual, all_pred_ci]).dropna()
    if all_values.empty:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = all_values.min(), all_values.max()
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        padding = (max_val - min_val) * 0.05
        min_val -= padding
        max_val += padding
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, lw=1, label="y=x")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")

    # --- Display Metrics Text ---
    if show_metrics and metrics_available:
        # (Logic for extracting and formatting metrics text remains the same)
        metrics_text = ""
        try:
            if can_plot_train:
                train_metrics = train_data[metric_cols].iloc[0]
                r2_tr = (
                    f"{train_metrics['split_R2']:.3f}"
                    if pd.notna(train_metrics["split_R2"])
                    else "N/A"
                )
                mae_tr = (
                    f"{train_metrics['split_MAE']:.3f}"
                    if pd.notna(train_metrics["split_MAE"])
                    else "N/A"
                )
                rmse_tr = (
                    f"{train_metrics['split_RMSE']:.3f}"
                    if pd.notna(train_metrics["split_RMSE"])
                    else "N/A"
                )
                metrics_text += f"Train: $R^2$={r2_tr}, MAE={mae_tr}, RMSE={rmse_tr}\n"
            else:
                metrics_text += "Train: N/A (No Data)\n"
            if can_plot_test:
                test_metrics = test_data[metric_cols].iloc[0]
                r2_te = (
                    f"{test_metrics['split_R2']:.3f}"
                    if pd.notna(test_metrics["split_R2"])
                    else "N/A"
                )
                mae_te = (
                    f"{test_metrics['split_MAE']:.3f}"
                    if pd.notna(test_metrics["split_MAE"])
                    else "N/A"
                )
                rmse_te = (
                    f"{test_metrics['split_RMSE']:.3f}"
                    if pd.notna(test_metrics["split_RMSE"])
                    else "N/A"
                )
                metrics_text += f"Test:  $R^2$={r2_te}, MAE={mae_te}, RMSE={rmse_te}"
            else:
                metrics_text += "Test:  N/A (No Data)"
            if metrics_text:
                ax.text(
                    0.05,
                    0.95,
                    metrics_text.strip(),
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                )
        except Exception as e:
            print(f"An error occurred during metrics display: {e}")
            ax.text(
                0.05,
                0.95,
                "Error displaying metrics",
                transform=ax.transAxes,
                fontsize=9,
                color="red",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            )


@op("LynxKite Graph Analytics", "plot qsar2", view="matplotlib")
def plot_qsar2(results_df: pd.DataFrame):
    """
    Plots actual vs. predicted values resembling the example image.

    Includes separate markers for train/test, y=x line, and parallel dashed
    error bands based on test set RMSE (optional). Does NOT use per-point CIs.

    Handles displaying the plot via plt.show() or saving it to a file
    based on the `save_path` parameter. THIS FUNCTION DOES NOT RETURN ANY VALUE.

    Parameters
    ----------
    results_df : pd.DataFrame
        Mandatory input DataFrame. Must contain: 'actual', 'predicted', 'split'.
        Should also contain 'split_RMSE' column for error bands and metrics display.
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional
    rmse_multiplier_for_bands : float or None, optional
        Determines the width of the dashed error bands (multiplier * test_RMSE).
        Set to None to disable bands. Default is 1.0.
    show_metrics : bool, optional
        Whether to display R2/MAE/RMSE text (requires metric columns). Default is True.
    save_path : str, optional
        If provided, saves plot to this path. If None (default), displays plot.

    Raises
    ------
    ValueError / TypeError : For invalid inputs.
    """
    COLOR_TRAIN = "royalblue"
    COLOR_TEST = "darkorange"  # Changed from red for potentially better contrast/appeal
    COLOR_PERFECT = "black"
    COLOR_BANDS = "dimgrey"  # Less prominent than the perfect line
    COLOR_GRID = "lightgrey"
    title = "QSAR Model Performance: Actual vs. Predicted"
    xlabel = "Actual Values"
    ylabel = "Predicted Values"
    # ci_alpha = 0.2
    show_metrics = True
    rmse_multiplier_for_bands = 1.0
    # --- Input Validation ---
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_cols = ["actual", "predicted", "split"]
    if not all(col in results_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    metric_cols = ["split_R2", "split_MAE", "split_RMSE"]
    metrics_available = all(col in results_df.columns for col in metric_cols)
    bands_possible = rmse_multiplier_for_bands is not None and "split_RMSE" in results_df.columns

    if show_metrics and not metrics_available:
        print(
            f"Warning: Metrics display requested, but one or more metric columns ({metric_cols}) are missing."
        )
    if rmse_multiplier_for_bands is not None and "split_RMSE" not in results_df.columns:
        print("Warning: Error bands requested, but 'split_RMSE' column is missing.")
        bands_possible = False

    # --- Prepare Data ---
    train_data = results_df[results_df["split"] == "train"].copy()
    test_data = results_df[results_df["split"] == "test"].copy()
    can_plot_train = not train_data.empty
    can_plot_test = not test_data.empty

    if not can_plot_train and not can_plot_test:
        print("Warning: Both training and test data subsets are empty. Cannot generate plot.")
        return

    # --- Create Plot with Style ---
    plt.style.use("seaborn-v0_8-whitegrid")  # Use a cleaner base style
    fig, ax = plt.subplots(figsize=(8, 8))  # Slightly larger figure

    # --- Plotting Logic ---
    # Scatter plots with enhanced style
    common_scatter_kws = {"s": 45, "alpha": 0.75, "edgecolor": "black", "linewidth": 0.5}
    if can_plot_train:
        ax.scatter(
            train_data["actual"],
            train_data["predicted"],
            label="Training set",
            marker="o",
            color=COLOR_TRAIN,
            **common_scatter_kws,
        )  # Blue circles

    if can_plot_test:
        ax.scatter(
            test_data["actual"],
            test_data["predicted"],
            label="Test set",
            marker="o",
            color=COLOR_TEST,
            **common_scatter_kws,
        )  # Orange circles

    # Determine plot limits
    # (Using the same logic as before to calculate min_val, max_val)
    all_actual = results_df["actual"].dropna()
    all_pred = results_df["predicted"].dropna()
    all_values = pd.concat([all_actual, all_pred]).dropna()
    if all_values.empty:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = all_values.min(), all_values.max()
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        data_range = max_val - min_val
        if data_range == 0:
            data_range = 1.0
        padding = data_range * 0.10
        min_val -= padding
        max_val += padding

    # Plot y=x line (Solid Black, slightly thicker)
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=COLOR_PERFECT,
        linestyle="-",
        linewidth=1.5,
        alpha=0.9,
        label="_nolegend_",
    )

    # Plot Error Bands based on Test RMSE (subtler style)
    rmse_test = np.nan
    if bands_possible and can_plot_test:
        try:
            rmse_test = test_data["split_RMSE"].dropna().iloc[0]
            if pd.notna(rmse_test) and rmse_test >= 0:
                margin = rmse_multiplier_for_bands * rmse_test
                band_label = (
                    f"$\pm {rmse_multiplier_for_bands}\,$RMSE"
                    if rmse_multiplier_for_bands == 1
                    else f"$\pm {rmse_multiplier_for_bands}\,$RMSE"
                )
                ax.plot(
                    [min_val, max_val],
                    [min_val + margin, max_val + margin],
                    color=COLOR_BANDS,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=band_label,
                )  # Grey dashed
                ax.plot(
                    [min_val, max_val],
                    [min_val - margin, max_val - margin],
                    color=COLOR_BANDS,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label="_nolegend_",
                )  # Grey dashed
            # else: print("Warning: Could not plot error bands (Invalid Test RMSE).") # Optionally silent
        except Exception as e:
            print(f"Warning: Could not plot error bands: {e}")

    # Set limits and aspect ratio
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    # ADD BACK Grid (Subtle Style)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color=COLOR_GRID, alpha=0.7)
    # Ensure grid is behind data points
    ax.set_axisbelow(True)

    # Set Labels and Title (using specified arguments)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15, pad=15, weight="semibold")  # Slightly larger title

    # Enhance Legend
    ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=10, shadow=False)

    # --- Display Metrics Text (Optional) ---
    if show_metrics and metrics_available:
        # (Logic for extracting and formatting metrics text remains the same)
        metrics_text = ""
        try:
            if can_plot_train:
                train_metrics = train_data[metric_cols].dropna().iloc[0]  # Ensure using valid row
                r2_tr = f"{train_metrics['split_R2']:.3f}"
                mae_tr = f"{train_metrics['split_MAE']:.3f}"
                rmse_tr = f"{train_metrics['split_RMSE']:.3f}"
                metrics_text += f"Train: $R^2$={r2_tr}, MAE={mae_tr}, RMSE={rmse_tr}\n"
            else:
                metrics_text += "Train: N/A\n"
            if can_plot_test:
                test_metrics = test_data[metric_cols].dropna().iloc[0]  # Ensure using valid row
                r2_te = f"{test_metrics['split_R2']:.3f}"
                mae_te = f"{test_metrics['split_MAE']:.3f}"
                rmse_te = f"{test_metrics['split_RMSE']:.3f}"
                metrics_text += f"Test:  $R^2$={r2_te}, MAE={mae_te}, RMSE={rmse_te}"
            else:
                metrics_text += "Test:  N/A"
            if metrics_text:
                ax.text(
                    0.05,
                    0.95,
                    metrics_text.strip(),
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )  # Adjusted box slightly
        except Exception as e:
            print(f"An error occurred during metrics display: {e}")
