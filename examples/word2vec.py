from lynxkite.core.ops import op
import staticvectors
import pandas as pd

ENV = "LynxKite Graph Analytics"


@op(ENV, "Word2vec for the top 1000 words", cache=True)
def word2vec_1000():
    model = staticvectors.StaticVectors("neuml/word2vec-quantized")
    with open("wordlist.txt") as f:
        words = [w.strip() for w in f.read().strip().split("\n")]
    df = pd.DataFrame(
        {
            "word": words,
            "embedding": model.embeddings(words).tolist(),
        }
    )
    return df


@op(ENV, "Take first N")
def first_n(df: pd.DataFrame, *, n=10):
    return df.head(n)
