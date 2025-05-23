"""LLM on Kubernetes, for use in bio projects or otherwise.

The BioNeMo NIMs are large. This module provides a small LLM that can be used
for demonstrating the Kubernetes lifecycle management without huge hardware
requirements.
"""

import openai
import pandas as pd
from lynxkite.core import ops

from . import k8s

ENV = "LynxKite Graph Analytics"
op = ops.op_registration(ENV)


@op("Ask LLM", slow=True)
@k8s.needs(
    name="lynxkite-bio-small-llm",
    image="vllm/vllm-openai:latest",
    port=8000,
    args=["--model", "google/gemma-3-1b-it"],
    health_probe="/health",
    forward_env=["HUGGING_FACE_HUB_TOKEN"],
    storage_path="/root/.cache/huggingface",
    storage_size="10Gi",
)
def ask_llm(df: pd.DataFrame, *, question: ops.LongStr, include_columns="<all>"):
    if not question:
        return df
    ip = k8s.get_ip("lynxkite-bio-small-llm")
    print(f"LLM is running at {ip}")
    client = openai.OpenAI(api_key="EMPTY", base_url=f"http://{ip}/v1")
    responses = []
    for row in df.iterrows():
        data = row[1].to_dict()
        if include_columns != "<all>":
            data = {k: v for k, v in data.items() if k in include_columns}
        prompt = (
            f"Answer the question based on the following data:\n\n{data}\n\nQuestion: {question}"
        )
        response = client.chat.completions.create(
            model="google/gemma-3-1b-it",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        responses.append(response.choices[0].message.content)
    df = df.copy()
    df["response"] = responses
    return df
