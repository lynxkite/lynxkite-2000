from lynxkite.core.ops import op
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem


@op("LynxKite Graph Analytics", "chembl sim search")
def similarity_to_dataframe(*, smiles: str, cutoff: int = 70) -> pd.DataFrame:
    """
    Run a ChEMBL similarity search and return the hits as a pandas DataFrame.
    If the SMILES is invalid or an error occurs, prints a message and returns
    an empty DataFrame with the expected columns.

    Parameters
    ----------
    smiles : str
        The SMILES string to search on.
    cutoff : int
        The minimum Tanimoto similarity (0–100).

    Returns
    -------
    pd.DataFrame
        Columns: 'molecule_chembl_id', 'similarity'
    """
    # Prepare empty frame to return on error
    cols = ["molecule_chembl_id", "similarity"]
    empty_df = pd.DataFrame(columns=cols)

    # 1) Quick SMILES validation
    if Chem.MolFromSmiles(smiles) is None:
        print("Please input a correct SMILES string.")
        return empty_df

    try:
        # 2) Do the ChEMBL API call
        similarity = new_client.similarity
        results = similarity.filter(smiles=smiles, similarity=cutoff).only(cols)

        # 3) Build DataFrame
        data = list(results)
        df = pd.DataFrame.from_records(data, columns=cols)

        # 4) Inform if no hits
        if df.empty:
            print("No hits found for that SMILES at the given cutoff.")
        return df

    except Exception as e:
        # Catch network errors, unexpected API replies, etc.
        print("An error occurred during the similarity search.")
        print("  Details:", str(e))
        return empty_df


@op("LynxKite Graph Analytics", "chembl structure")
def _chembl_structures(
    df: pd.DataFrame, *, id_col: str = "molecule_chembl_id", timeout: int = 5
) -> pd.DataFrame:
    """
    Given a DataFrame with a column of ChEMBL molecule IDs, append
    canonical SMILES, standard InChI, and standard InChIKey.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame; must contain `id_col`.
    id_col : str
        Name of the column in `df` that holds ChEMBL IDs (e.g. 'CHEMBL1234').
    timeout : int
        How many seconds to wait for the API (not currently used by chembl client,
        but reserved for future enhancements or custom wrappers).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with three additional columns:
          - smiles
          - standard_inchi
          - standard_inchi_key
    """
    # make a copy so we don’t modify in-place
    out = df.copy()
    # prepare new columns
    out["smiles"] = None
    out["standard_inchi"] = None
    out["standard_inchi_key"] = None

    mol_client = new_client.molecule

    for idx, chembl_id in out[id_col].items():
        try:
            # query ChEMBL for this molecule
            res = mol_client.filter(chembl_id=chembl_id).only(
                ["molecule_chembl_id", "molecule_structures"]
            )
            # filter() returns an iterable; grab first record if exists
            rec = next(iter(res), None)
            if rec and rec.get("molecule_structures"):
                struct = rec["molecule_structures"]
                out.at[idx, "smiles"] = struct.get("canonical_smiles")
                out.at[idx, "standard_inchi"] = struct.get("standard_inchi")
                out.at[idx, "standard_inchi_key"] = struct.get("standard_inchi_key")
            else:
                print(f"[Warning] No structure found for {chembl_id}")
        except Exception as e:
            print(f"[Error] Lookup failed for {chembl_id}: {e!s}")

    return out


@op("LynxKite Graph Analytics", "get chembl drugs")
def fetch_chembl_drugs(
    *, first_approval: int = 2000, development_phase: int = None
) -> pd.DataFrame:
    """
    Fetch drugs from ChEMBL matching the given USAN stem, approval year,
    and development phase, returning key fields as a DataFrame.

    Parameters
    ----------
    first_approval : int, optional
        Only include drugs first approved in or after this year (default=1980).
    development_phase : int, optional
        Only include drugs in this development phase (e.g. 2, 3, 4).
        If None, do not filter by phase.
    usan_stem : str, optional
        USAN stem to filter on (default="-azosin").

    Returns
    -------
    pd.DataFrame
        Columns:
          - development_phase
          - first_approval
          - molecule_chembl_id
          - synonyms
          - usan_stem
          - usan_stem_definition
          - usan_year

        If no results (or on error), returns an empty DataFrame with these columns.
    """
    cols = [
        "development_phase",
        "first_approval",
        "molecule_chembl_id",
        "synonyms",
        "usan_stem",
        "usan_stem_definition",
        "usan_year",
    ]
    empty_df = pd.DataFrame(columns=cols)

    # Validate inputs
    if first_approval is not None and not isinstance(first_approval, int):
        print("Error: first_approval must be an integer year.")
        return empty_df
    if development_phase is not None and not isinstance(development_phase, int):
        print("Error: development_phase must be an integer.")
        return empty_df
    # if not isinstance(usan_stem, str):
    #     print("Error: usan_stem must be a string.")
    #     return empty_df

    try:
        drug = new_client.drug

        # apply approval-year filter
        if first_approval is not None:
            drug = drug.filter(first_approval__gte=first_approval)
        # apply development-phase filter
        if development_phase is not None:
            drug = drug.filter(development_phase=development_phase)
        # apply USAN stem filter
        # drug = drug.filter(usan_stem=usan_stem)

        res = drug.only(cols)
        df = pd.DataFrame(res, columns=cols)

        if df.empty:
            print("No drugs found for those filters.")
        return df

    except Exception as e:
        print("An error occurred during the ChEMBL query:")
        print(" ", str(e))
        return empty_df


@op("LynxKite Graph Analytics", "get bioactivity from uniprot")
def fetch_chembl_bioactivity(*, uniprot_id: str = "Q9NZQ7"):
    """
    Fetch bioactivity data from ChEMBL for a given UniProt ID.
    """
    target = new_client.target.filter(target_components__accession=uniprot_id)
    targets = list(target)
    if not targets:
        return []

    target_chembl_id = targets[0]["target_chembl_id"]
    activities = new_client.activity.filter(
        target_chembl_id=target_chembl_id, standard_type__in=["IC50", "Ki", "Kd"]
    )
    df = pd.DataFrame(activities)
    return df
