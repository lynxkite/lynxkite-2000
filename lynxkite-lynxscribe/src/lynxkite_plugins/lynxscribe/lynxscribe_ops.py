"""
LynxScribe configuration and testing in LynxKite.
"""

from lynxscribe.core.llm.base import get_llm_engine
from lynxscribe.core.vector_store.base import get_vector_store
from lynxscribe.common.config import load_config
from lynxscribe.components.text.embedder import TextEmbedder
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

ENV = "LynxScribe"
one_by_one.register(ENV)
op = ops.op_registration(ENV)
output_on_top = ops.output_position(output="top")


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
