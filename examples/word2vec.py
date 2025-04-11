from lynxkite.core.ops import op
import staticvectors
import pandas as pd

ENV = "LynxKite Graph Analytics"


@op(ENV, "Word2vec for the top 1000 words", slow=True)
def word2vec_1000():
    model = staticvectors.StaticVectors("neuml/word2vec-quantized")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/deekayen/4148741/raw/98d35708fa344717d8eee15d11987de6c8e26d7d/1-1000.txt",
        names=["word"],
    )
    df["embedding"] = model.embeddings(df.word.tolist()).tolist()
    return df


@op(ENV, "Take first N")
def first_n(df: pd.DataFrame, *, n=10):
    return df.head(n)


@op(ENV, "Sample N")
def sample_n(df: pd.DataFrame, *, n=10):
    return df.sample(n)
