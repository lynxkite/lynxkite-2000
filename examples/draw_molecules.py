from lynxkite.core.ops import op
import pandas as pd
import base64
import io


def pil_to_data(image):
    buffer = io.BytesIO()
    image.save(buffer, format="png")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


def smiles_to_data(smiles):
    import rdkit

    m = rdkit.Chem.MolFromSmiles(smiles)
    img = rdkit.Chem.Draw.MolToImage(m)
    data = pil_to_data(img)
    return data


@op("LynxKite Graph Analytics", "Draw molecules")
def draw_molecules(df: pd.DataFrame, *, smiles_column: str, image_column: str = "image"):
    df = df.copy()
    df[image_column] = df[smiles_column].apply(smiles_to_data)
    return df
