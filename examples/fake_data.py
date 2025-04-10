from lynxkite.core.ops import op
from faker import Faker
import pandas as pd

faker = Faker()


@op("LynxKite Graph Analytics", "Fake data")
def fake(*, n=10):
    df = pd.DataFrame(
        {
            "name": [faker.name() for _ in range(n)],
            "address": [faker.address() for _ in range(n)],
        }
    )
    return df
