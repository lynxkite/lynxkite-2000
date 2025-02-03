"""For specifying an LLM agent logic flow.

This is very much a prototype. It might end up merged into LynxScribe
as an "agentic logic flow". It might just get deleted.

(This is why the dependencies are left hanging.)
"""

from lynxkite.core import ops
import enum
import jinja2
import json
import numpy as np
import pandas as pd
from lynxkite.core.executors import one_by_one

jinja = jinja2.Environment()
chroma_client = None
LLM_CACHE = {}
ENV = "LLM logic"
one_by_one.register(ENV)
op = ops.op_registration(ENV)


def chat(*args, **kwargs):
    import openai

    chat_client = openai.OpenAI(base_url="http://localhost:8080/v1")
    key = json.dumps({"method": "chat", "args": args, "kwargs": kwargs})
    if key not in LLM_CACHE:
        completion = chat_client.chat.completions.create(*args, **kwargs)
        LLM_CACHE[key] = [c.message.content for c in completion.choices]
    return LLM_CACHE[key]


def embedding(*args, **kwargs):
    import openai

    embedding_client = openai.OpenAI(base_url="http://localhost:7997/")
    key = json.dumps({"method": "embedding", "args": args, "kwargs": kwargs})
    if key not in LLM_CACHE:
        res = embedding_client.embeddings.create(*args, **kwargs)
        [data] = res.data
        LLM_CACHE[key] = data.embedding
    return LLM_CACHE[key]


@op("Input CSV")
def input_csv(*, filename: ops.PathStr, key: str):
    return pd.read_csv(filename).rename(columns={key: "text"})


@op("Input document")
def input_document(*, filename: ops.PathStr):
    with open(filename) as f:
        return {"text": f.read()}


@op("Input chat")
def input_chat(*, chat: str):
    return {"text": chat}


@op("Split document")
def split_document(input, *, delimiter: str = "\\n\\n"):
    delimiter = delimiter.encode().decode("unicode_escape")
    chunks = input["text"].split(delimiter)
    return pd.DataFrame(chunks, columns=["text"])


@ops.input_side(input=ops.Side.TOP)
@op("Build document graph")
def build_document_graph(input):
    return [{"source": i, "target": i + 1} for i in range(len(input) - 1)]


@ops.input_side(nodes=ops.Side.TOP, edges=ops.Side.TOP)
@op("Predict links")
def predict_links(nodes, edges):
    """A placeholder for a real algorithm. For now just adds 2-hop neighbors."""
    edge_map = {}  # Source -> [Targets]
    for edge in edges:
        edge_map.setdefault(edge["source"], [])
        edge_map[edge["source"]].append(edge["target"])
    new_edges = []
    for edge in edges:
        for t in edge_map.get(edge["target"], []):
            new_edges.append({"source": edge["source"], "target": t})
    return edges + new_edges


@ops.input_side(nodes=ops.Side.TOP, edges=ops.Side.TOP)
@op("Add neighbors")
def add_neighbors(nodes, edges, item):
    nodes = pd.DataFrame(nodes)
    edges = pd.DataFrame(edges)
    matches = item["rag"]
    additional_matches = []
    for m in matches:
        node = nodes[nodes["text"] == m].index[0]
        neighbors = edges[edges["source"] == node]["target"].to_list()
        additional_matches.extend(nodes.loc[neighbors, "text"])
    return {**item, "rag": matches + additional_matches}


@op("Create prompt")
def create_prompt(input, *, save_as="prompt", template: ops.LongStr):
    assert template, (
        "Please specify the template. Refer to columns using the Jinja2 syntax."
    )
    t = jinja.from_string(template)
    prompt = t.render(**input)
    return {**input, save_as: prompt}


@op("Ask LLM")
def ask_llm(input, *, model: str, accepted_regex: str = None, max_tokens: int = 100):
    assert model, "Please specify the model."
    assert "prompt" in input, "Please create the prompt first."
    options = {}
    if accepted_regex:
        options["extra_body"] = {
            "guided_regex": accepted_regex,
        }
    results = chat(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": input["prompt"]},
        ],
        **options,
    )
    return [{**input, "response": r} for r in results]


@op("View", view=ops.ViewType.TABLE_VIEW)
def view(input, *, _ctx: one_by_one.Context):
    v = _ctx.last_result
    if v:
        columns = v["dataframes"]["df"]["columns"]
        v["dataframes"]["df"]["data"].append([input[c] for c in columns])
    else:
        columns = [str(c) for c in input.keys() if not str(c).startswith("_")]
        v = {
            "dataframes": {
                "df": {
                    "columns": columns,
                    "data": [[input[c] for c in columns]],
                }
            }
        }
    return v


@ops.input_side(input=ops.Side.RIGHT)
@ops.output_side(output=ops.Side.LEFT)
@op("Loop")
def loop(input, *, max_iterations: int = 3, _ctx: one_by_one.Context):
    """Data can flow back here max_iterations-1 times."""
    key = f"iterations-{_ctx.node.id}"
    input[key] = input.get(key, 0) + 1
    if input[key] < max_iterations:
        return input


@op("Branch", outputs=["true", "false"])
def branch(input, *, expression: str):
    res = eval(expression, input)
    return one_by_one.Output(output_handle=str(bool(res)).lower(), value=input)


class RagEngine(enum.Enum):
    Chroma = "Chroma"
    Custom = "Custom"


@ops.input_side(db=ops.Side.TOP)
@op("RAG")
def rag(
    input,
    db,
    *,
    engine: RagEngine = RagEngine.Chroma,
    input_field="text",
    db_field="text",
    num_matches: int = 10,
    _ctx: one_by_one.Context,
):
    global chroma_client
    if engine == RagEngine.Chroma:
        last = _ctx.last_result
        if last:
            collection = last["_collection"]
        else:
            collection_name = _ctx.node.id.replace(" ", "_")
            if chroma_client is None:
                import chromadb

                chroma_client = chromadb.Client()
            for c in chroma_client.list_collections():
                if c.name == collection_name:
                    chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(name=collection_name)
            collection.add(
                documents=[r[db_field] for r in db],
                ids=[str(i) for i in range(len(db))],
            )
        results = collection.query(
            query_texts=[input[input_field]],
            n_results=num_matches,
        )
        results = [db[int(r)] for r in results["ids"][0]]
        return {**input, "rag": results, "_collection": collection}
    if engine == RagEngine.Custom:
        model = "google/gemma-2-2b-it"
        chat = input[input_field]
        embeddings = [embedding(input=[r[db_field]], model=model) for r in db]
        q = embedding(input=[chat], model=model)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scores = [(i, cosine_similarity(q, e)) for i, e in enumerate(embeddings)]
        scores.sort(key=lambda x: -x[1])
        matches = [db[i][db_field] for i, _ in scores[:num_matches]]
        return {**input, "rag": matches}


@op("Run Python")
def run_python(input, *, template: str):
    """TODO: Implement."""
    return input
