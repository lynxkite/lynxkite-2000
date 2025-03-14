"""
LynxScribe configuration and testing in LynxKite.
"""

from google.cloud import storage
from copy import deepcopy
import asyncio
import pandas as pd
import os
import joblib

from lynxscribe.core.llm.base import get_llm_engine
from lynxscribe.core.vector_store.base import get_vector_store
from lynxscribe.common.config import load_config
from lynxscribe.components.text.embedder import TextEmbedder
from lynxscribe.core.models.embedding import Embedding

from lynxscribe.components.rag.rag_graph import RAGGraph
from lynxscribe.components.rag.knowledge_base_graph import PandasKnowledgeBaseGraph
from lynxscribe.components.rag.rag_chatbot import Scenario, ScenarioSelector, RAGChatbot
from lynxscribe.components.chat.processors import (
    ChatProcessor,
    MaskTemplate,
    TruncateHistory,
)
from lynxscribe.components.chat.api import ChatAPI
from lynxscribe.core.models.prompts import ChatCompletionPrompt

from lynxkite.core import ops
import json
from lynxkite.core.executors import one_by_one

# logger
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

ENV = "LynxScribe"
one_by_one.register(ENV)
os.makedirs("../joblib-cache", exist_ok=True)
mem = joblib.Memory("../joblib-cache")
op = ops.op_registration(ENV)
output_on_top = ops.output_position(output="top")


@op("GCP Image Loader")
def gcp_image_loader(
    *,
    gcp_bucket: str = "lynxkite_public_data",
    prefix: str = "lynxscribe-images/image-rag-test",
):
    """
    Gives back the list of URLs of all the images in the GCP storage.
    """
    client = storage.Client()
    bucket = client.bucket(gcp_bucket)
    blobs = bucket.list_blobs(prefix=prefix)
    image_urls = [
        blob.public_url
        for blob in blobs
        if blob.name.endswith((".jpg", ".jpeg", ".png"))
    ]
    return {"image_urls": image_urls}


@output_on_top
@op("LynxScribe RAG Vector Store")
# @mem.cache
def ls_rag_graph(
    *,
    name: str = "faiss",
    num_dimensions: int = 3072,
    collection_name: str = "lynx",
    text_embedder_interface: str = "openai",
    text_embedder_model_name_or_path: str = "text-embedding-3-large",
    api_key_name: str = "OPENAI_API_KEY",
):
    """
    Returns with a vector store instance.
    """

    # getting the text embedder instance
    llm_params = {"name": text_embedder_interface}
    if api_key_name:
        llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)
    text_embedder = TextEmbedder(llm=llm, model=text_embedder_model_name_or_path)

    # getting the vector store
    if name == "chromadb":
        vector_store = get_vector_store(name=name, collection_name=collection_name)
    elif name == "faiss":
        vector_store = get_vector_store(name=name, num_dimensions=num_dimensions)
    else:
        raise ValueError(f"Vector store name '{name}' is not supported.")

    # building up the RAG graph
    rag_graph = RAGGraph(
        PandasKnowledgeBaseGraph(vector_store=vector_store, text_embedder=text_embedder)
    )

    return {"rag_graph": rag_graph}


@output_on_top
@op("LynxScribe Image Describer")
# @mem.cache
def ls_image_describer(
    *,
    llm_interface: str = "openai",
    llm_visual_model: str = "gpt-4o",
    llm_prompt_path: str = "lynxkite-lynxscribe/promptdb/image_description_prompts.yaml",
    llm_prompt_name: str = "cot_picture_descriptor",
    api_key_name: str = "OPENAI_API_KEY",
):
    """
    Returns with an image describer instance.
    TODO: adding a relative path to the prompt path + adding model kwargs
    """

    llm_params = {"name": llm_interface}
    if api_key_name:
        llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)

    prompt_base = load_config(llm_prompt_path)[llm_prompt_name]

    return {
        "image_describer": {
            "llm": llm,
            "prompt_base": prompt_base,
            "model": llm_visual_model,
        }
    }


@ops.input_position(image_describer="bottom", rag_graph="bottom")
@op("LynxScribe Image RAG Builder")
# @mem.cache
async def ls_image_rag_builder(
    image_urls,
    image_describer,
    rag_graph,
    *,
    image_rag_out_path: str = "image_test_rag_graph.pickle",
):
    """
    Based on an input image folder (currently only supports GCP storage),
    the function builds up an image RAG graph, where the nodes are the
    descriptions of the images (and of all image objects).

    In a later phase, synthetic questions and "named entities" will also
    be added to the graph.
    """

    # handling inputs
    image_describer = image_describer[0]["image_describer"]
    image_urls = image_urls["image_urls"]
    rag_graph = rag_graph[0]["rag_graph"]

    # generate prompts from inputs
    prompt_list = []
    for i in range(len(image_urls)):
        image = image_urls[i]

        _prompt = deepcopy(image_describer["prompt_base"])
        for message in _prompt:
            if isinstance(message["content"], list):
                for _message_part in message["content"]:
                    if "image_url" in _message_part:
                        _message_part["image_url"] = {"url": image}

        prompt_list.append(_prompt)
    ch_prompt_list = [
        ChatCompletionPrompt(model=image_describer["model"], messages=prompt)
        for prompt in prompt_list
    ]

    # get the image descriptions
    llm = image_describer["llm"]
    tasks = [
        llm.acreate_completion(completion_prompt=_prompt) for _prompt in ch_prompt_list
    ]
    out_completions = await asyncio.gather(*tasks)
    results = [
        dictionary_corrector(result.choices[0].message.content)
        for result in out_completions
    ]

    # generate combination of descriptions and embed them
    text_embedder = rag_graph.kg_base.text_embedder

    dict_list_df = []
    for _i, _result in enumerate(results):
        url_res = image_urls[_i]

        if "overall description" in _result:
            dict_list_df.append(
                {
                    "image_url": url_res,
                    "description": _result["overall description"],
                    "source": "overall description",
                }
            )

        if "details" in _result:
            for dkey in _result["details"].keys():
                text = f"The picture's description is: {_result['overall description']}\n\nThe description of the {dkey} is: {_result['details'][dkey]}"
                dict_list_df.append(
                    {"image_url": url_res, "description": text, "source": "details"}
                )

    pdf_descriptions = pd.DataFrame(dict_list_df)
    pdf_descriptions["embedding_values"] = await text_embedder.acreate_embedding(
        pdf_descriptions["description"].to_list()
    )
    pdf_descriptions["id"] = "im_" + pdf_descriptions.index.astype(str)

    # adding the embeddings to the RAG graph with metadata
    pdf_descriptions["embedding"] = pdf_descriptions.apply(
        lambda row: Embedding(
            id=row["id"],
            value=row["embedding_values"],
            metadata={
                "image_url": row["image_url"],
                "image_part": row["source"],
                "type": "image_description",
            },
            document=row["description"],
        ),
        axis=1,
    )
    embedding_list = pdf_descriptions["embedding"].tolist()

    # adding the embeddings to the RAG graph
    rag_graph.kg_base.vector_store.upsert(embedding_list)

    # # saving the RAG graph
    # rag_graph.kg_base.save(image_rag_out_path)

    return {"knowledge_base": rag_graph}


@op("LynxScribe RAG Graph Saver")
def ls_save_rag_graph(
    knowledge_base,
    *,
    image_rag_out_path: str = "image_test_rag_graph.pickle",
):
    """
    Saves the RAG graph to a pickle file.
    """

    knowledge_base.kg_base.save(image_rag_out_path)
    return None


@ops.input_position(rag_graph="bottom")
@op("LynxScribe Image RAG Query")
async def search_context(rag_graph, text, *, top_k=3):
    message = text["text"]
    rag_graph = rag_graph[0]["knowledge_base"]

    # get all similarities
    emb_similarities = await rag_graph.search_context(
        message, max_results=top_k, unique_metadata_key="image_url"
    )

    # get the image urls, scores and descriptions
    result_list = []

    for emb_sim in emb_similarities:
        image_url = emb_sim.embedding.metadata["image_url"]
        score = emb_sim.score
        description = emb_sim.embedding.document
        result_list.append(
            {"image_url": image_url, "score": score, "description": description}
        )

    print(result_list)
    return {"embedding_similarities": result_list}


@op("View image", view="image")
def view_image(embedding_similarities):
    """
    Plotting the selected image.
    """
    embedding_similarities = embedding_similarities["embedding_similarities"]
    return embedding_similarities[0]["image_url"]


@output_on_top
@op("Vector store")
def vector_store(*, name="chromadb", collection_name="lynx"):
    vector_store = get_vector_store(name=name, collection_name=collection_name)
    return {"vector_store": vector_store}


@output_on_top
@op("LLM")
def llm(*, name="openai"):
    llm = get_llm_engine(name=name)
    return {"llm": llm}


@output_on_top
@ops.input_position(llm="bottom")
@op("Text embedder")
def text_embedder(llm, *, model="text-embedding-ada-002"):
    llm = llm[0]["llm"]
    text_embedder = TextEmbedder(llm=llm, model=model)
    return {"text_embedder": text_embedder}


@output_on_top
@ops.input_position(vector_store="bottom", text_embedder="bottom")
@op("RAG graph")
def rag_graph(vector_store, text_embedder):
    vector_store = vector_store[0]["vector_store"]
    text_embedder = text_embedder[0]["text_embedder"]
    rag_graph = RAGGraph(
        PandasKnowledgeBaseGraph(vector_store=vector_store, text_embedder=text_embedder)
    )
    return {"rag_graph": rag_graph}


@output_on_top
@op("Scenario selector")
def scenario_selector(*, scenario_file: str, node_types="intent_cluster"):
    scenarios = load_config(scenario_file)
    node_types = [t.strip() for t in node_types.split(",")]
    scenario_selector = ScenarioSelector(
        scenarios=[Scenario(**scenario) for scenario in scenarios],
        node_types=node_types,
    )
    return {"scenario_selector": scenario_selector}


DEFAULT_NEGATIVE_ANSWER = "I'm sorry, but the data I've been trained on does not contain any information related to your question."


@output_on_top
@ops.input_position(rag_graph="bottom", scenario_selector="bottom", llm="bottom")
@op("RAG chatbot")
def rag_chatbot(
    rag_graph,
    scenario_selector,
    llm,
    *,
    negative_answer=DEFAULT_NEGATIVE_ANSWER,
    limits_by_type="{}",
    strict_limits=True,
    max_results=5,
):
    rag_graph = rag_graph[0]["rag_graph"]
    scenario_selector = scenario_selector[0]["scenario_selector"]
    llm = llm[0]["llm"]
    limits_by_type = json.loads(limits_by_type)
    rag_chatbot = RAGChatbot(
        rag_graph=rag_graph,
        scenario_selector=scenario_selector,
        llm=llm,
        negative_answer=negative_answer,
        limits_by_type=limits_by_type,
        strict_limits=strict_limits,
        max_results=max_results,
    )
    return {"chatbot": rag_chatbot}


@output_on_top
@ops.input_position(processor="bottom")
@op("Chat processor")
def chat_processor(processor, *, _ctx: one_by_one.Context):
    cfg = _ctx.last_result or {
        "question_processors": [],
        "answer_processors": [],
        "masks": [],
    }
    for f in ["question_processor", "answer_processor", "mask"]:
        if f in processor:
            cfg[f + "s"].append(processor[f])
    question_processors = cfg["question_processors"][:]
    answer_processors = cfg["answer_processors"][:]
    masking_templates = {}
    for mask in cfg["masks"]:
        masking_templates[mask["name"]] = mask
    if masking_templates:
        question_processors.append(MaskTemplate(masking_templates=masking_templates))
        answer_processors.append(MaskTemplate(masking_templates=masking_templates))
    chat_processor = ChatProcessor(
        question_processors=question_processors, answer_processors=answer_processors
    )
    return {"chat_processor": chat_processor, **cfg}


@output_on_top
@op("Truncate history")
def truncate_history(*, max_tokens=10000):
    return {"question_processor": TruncateHistory(max_tokens=max_tokens)}


@output_on_top
@op("Mask")
def mask(*, name="", regex="", exceptions="", mask_pattern=""):
    exceptions = [e.strip() for e in exceptions.split(",") if e.strip()]
    return {
        "mask": {
            "name": name,
            "regex": regex,
            "exceptions": exceptions,
            "mask_pattern": mask_pattern,
        }
    }


@ops.input_position(chat_api="bottom")
@op("Test Chat API")
async def test_chat_api(message, chat_api, *, show_details=False):
    chat_api = chat_api[0]["chat_api"]
    request = ChatCompletionPrompt(
        model="",
        messages=[{"role": "user", "content": message["text"]}],
    )
    response = await chat_api.answer(request, stream=False)
    answer = response.choices[0].message.content
    if show_details:
        return {"answer": answer, **response.__dict__}
    else:
        return {"answer": answer}


@op("Input chat")
def input_chat(*, chat: str):
    return {"text": chat}


@output_on_top
@ops.input_position(chatbot="bottom", chat_processor="bottom", knowledge_base="bottom")
@op("Chat API")
def chat_api(chatbot, chat_processor, knowledge_base, *, model="gpt-4o-mini"):
    chatbot = chatbot[0]["chatbot"]
    chat_processor = chat_processor[0]["chat_processor"]
    knowledge_base = knowledge_base[0]
    c = ChatAPI(
        chatbot=chatbot,
        chat_processor=chat_processor,
        model=model,
    )
    if knowledge_base:
        c.chatbot.rag_graph.kg_base.load_v1_knowledge_base(**knowledge_base)
        c.chatbot.scenario_selector.check_compatibility(c.chatbot.rag_graph)
    return {"chat_api": c}


@output_on_top
@op("Knowledge base")
def knowledge_base(
    *,
    nodes_path="nodes.pickle",
    edges_path="edges.pickle",
    template_cluster_path="tempclusters.pickle",
):
    return {
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "template_cluster_path": template_cluster_path,
    }


@op("View", view="table_view")
def view(input):
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


async def get_chat_api(ws):
    import pathlib
    from lynxkite.core import workspace

    DATA_PATH = pathlib.Path.cwd() / "data"
    path = DATA_PATH / ws
    assert path.is_relative_to(DATA_PATH)
    assert path.exists(), f"Workspace {path} does not exist"
    ws = workspace.load(path)
    contexts = await ops.EXECUTORS[ENV](ws)
    nodes = [op for op in ws.nodes if op.data.title == "Chat API"]
    [node] = nodes
    context = contexts[node.id]
    return context.last_result["chat_api"]


async def stream_chat_api_response(request):
    chat_api = await get_chat_api(request["model"])
    request = ChatCompletionPrompt(**request)
    async for chunk in await chat_api.answer(request, stream=True):
        yield chunk.model_dump_json()


async def api_service_post(request):
    """
    Serves a chat endpoint that matches LynxScribe's interface.
    To access it you need to add the "module" and "workspace"
    parameters.
    The workspace must contain exactly one "Chat API" node.

      curl -X POST ${LYNXKITE_URL}/api/service/server.lynxkite_ops \
        -H "Content-Type: application/json" \
        -d '{
          "model": "LynxScribe demo",
          "messages": [{"role": "user", "content": "what does the fox say"}]
        }'
    """
    path = "/".join(request.url.path.split("/")[4:])
    request = await request.json()
    if path == "chat/completions":
        from sse_starlette.sse import EventSourceResponse

        return EventSourceResponse(stream_chat_api_response(request))
    return {"error": "Not found"}


async def api_service_get(request):
    path = "/".join(request.url.path.split("/")[4:])
    if path == "models":
        return {
            "object": "list",
            "data": [
                {
                    "id": ws,
                    "object": "model",
                    "created": 0,
                    "owned_by": "lynxkite",
                    "meta": {"profile_image_url": "https://lynxkite.com/favicon.png"},
                }
                for ws in get_lynxscribe_workspaces()
            ],
        }
    return {"error": "Not found"}


def get_lynxscribe_workspaces():
    import pathlib
    from lynxkite.core import workspace

    DATA_DIR = pathlib.Path.cwd() / "data"
    workspaces = []
    for p in DATA_DIR.glob("**/*"):
        if p.is_file():
            try:
                ws = workspace.load(p)
                if ws.env == ENV:
                    workspaces.append(p.relative_to(DATA_DIR))
            except Exception:
                pass  # Ignore files that are not valid workspaces.
    workspaces.sort()
    return workspaces


def dictionary_corrector(dict_string: str, expected_keys: list | None = None) -> dict:
    """
    Processing LLM outputs: when the LLM returns with a dictionary (in a string format). It optionally
    crosschecks the input with the expected keys and return a dictionary with the expected keys and their
    values ('unknown' if not present). If there is an error during the processing, it will return with
    a dictionary of the expected keys, all with 'error' as a value (or with an empty dictionary).

    Currently the function does not delete the extra key-value pairs.
    """

    out_dict = {}

    if len(dict_string) == 0:
        return out_dict

    # deleting the optional text before the first and after the last curly brackets
    dstring_prc = dict_string
    if dstring_prc[0] != "{":
        dstring_prc = "{" + "{".join(dstring_prc.split("{")[1:])
    if dstring_prc[-1] != "}":
        dstring_prc = "}".join(dstring_prc.split("}")[:-1]) + "}"

    try:
        trf_dict = eval(dstring_prc)
        if expected_keys:
            for _key in expected_keys:
                if _key in trf_dict:
                    out_dict[_key] = trf_dict[_key]
                else:
                    out_dict[_key] = "unknown"
        else:
            out_dict = trf_dict
    except Exception:
        if expected_keys:
            for _key in expected_keys:
                out_dict[_key] = "error"

    return out_dict
