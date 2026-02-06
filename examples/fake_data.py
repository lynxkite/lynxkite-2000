from lynxkite_core.ops import op
from faker import Faker  # ty: ignore
import pandas as pd

faker = Faker()


@op("LynxKite Graph Analytics", "Fake data")
def fake(*, n=10):
    """Creates a DataFrame with random-generated names and postal addresses.

    Parameters:
        n: Number of rows to create.
    """
    df = pd.DataFrame(
        {
            "name": [faker.name() for _ in range(n)],
            "address": [faker.address() for _ in range(n)],
        }
    )
    return df
