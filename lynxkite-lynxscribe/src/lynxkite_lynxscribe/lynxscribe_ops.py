"""
LynxScribe configuration and testing in LynxKite.
TODO: all these outputs should contain metadata. So the next task can check the input type, etc.
"""

from google.cloud import storage
from copy import deepcopy
from enum import Enum
import asyncio
import pandas as pd
import joblib
from pydantic import BaseModel, ConfigDict

import pathlib
from lynxscribe.core.llm.base import get_llm_engine
from lynxscribe.core.vector_store.base import get_vector_store
from lynxscribe.common.config import load_config
from lynxscribe.components.text.embedder import TextEmbedder
from lynxscribe.core.models.embedding import Embedding
from lynxscribe.components.embedding_clustering import FclusterBasedClustering

from lynxscribe.components.rag.rag_graph import RAGGraph
from lynxscribe.components.rag.knowledge_base_graph import PandasKnowledgeBaseGraph
from lynxscribe.components.rag.rag_chatbot import Scenario, ScenarioSelector, RAGChatbot
from lynxscribe.components.chat.processors import (
    ChatProcessor,
    MaskTemplate,
    TruncateHistory,
)
from lynxscribe.components.chat.api import ChatAPI
from lynxscribe.core.models.prompts import ChatCompletionPrompt, Message
from lynxscribe.components.rag.loaders import FAQTemplateLoader

from lynxkite.core import ops
import json
from lynxkite.core.executors import one_by_one

DEFAULT_NEGATIVE_ANSWER = "I'm sorry, but the data I've been trained on does not contain any information related to your question."

ENV = "LynxScribe"
one_by_one.register(ENV)
mem = joblib.Memory("joblib-cache")
op = ops.op_registration(ENV)
output_on_top = ops.output_position(output="top")


# defining the cloud provider enum
class CloudProvider(Enum):
    GCP = "gcp"
    AWS = "aws"
    AZURE = "azure"


class RAGVersion(Enum):
    V1 = "v1"
    V2 = "v2"


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"


class RAGTemplate(BaseModel):
    """
    Model for RAG templates consisting of three tables: they are connected via scenario names.
    One table (FAQs) contains scenario-denoted nodes to upsert into the knowledge base, the other
    two tables serve as the configuration for the scenario selector.
    Attributes:
        faq_data:
            Table where each row is an FAQ question, and possibly its answer pair. Will be fed into
            `FAQTemplateLoader.load_nodes_and_edges()`. For configuration of this table see the
            loader's init arguments.
        scenario_data:
            Table where each row is a Scenario, column names are thus scenario attributes. Will be
            fed into `ScenarioSelector.from_data()`.
        prompt_codes:
            Optional helper for the scenario table, may contain prompt code mappings to real prompt
            messages. It's enough then to use the codes instead of the full messages in the
            scenarios table. Will be fed into `ScenarioSelector.from_data()`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    faq_data: pd.DataFrame
    scenario_data: pd.DataFrame
    prompt_codes: dict[str, str] = {}

    @classmethod
    def from_excel_path(
        cls,
        path: str,
        faq_data_sheet_name: str,
        scenario_data_sheet_name: str,
        prompt_codes_sheet_name: str | None = None,
    ) -> "RAGTemplate":
        """Spawn a RAGTemplate from an Excel file containing the two needed (plus one optional) sheets."""

        def transform_codes(prompt_codes: pd.DataFrame) -> dict[str, str]:
            """Check and transform prompt codes table into a code dictionary."""
            if (len_columns := len(prompt_codes.columns)) != 2:
                raise ValueError(
                    f"Prompt codes should contain exactly 2 columns, {len_columns} found."
                )
            return prompt_codes.set_index(prompt_codes.columns[0])[
                prompt_codes.columns[1]
            ].to_dict()

        return cls(
            faq_data=pd.read_excel(path, sheet_name=faq_data_sheet_name),
            scenario_data=pd.read_excel(path, sheet_name=scenario_data_sheet_name),
            prompt_codes=transform_codes(pd.read_excel(path, sheet_name=prompt_codes_sheet_name))
            if prompt_codes_sheet_name
            else {},
        )


@op("Cloud-sourced File Listing")
def cloud_file_loader(
    *,
    cloud_provider: CloudProvider = CloudProvider.GCP,
    folder_URL: str = "https://storage.googleapis.com/lynxkite_public_data/lynxscribe-images/image-rag-test",
    accepted_file_types: str = ".jpg, .jpeg, .png",
):
    """
    Gives back the list of URLs of all the images from a cloud-based folder.
    Currently only supports GCP storage.
    """
    if folder_URL[-1].endswith("/"):
        folder_URL = folder_URL[:-1]

    accepted_file_types = tuple([t.strip() for t in accepted_file_types.split(",")])

    if cloud_provider == CloudProvider.GCP:
        client = storage.Client()
        url_useful_part = folder_URL.split(".com/")[-1]
        bucket_name = url_useful_part.split("/")[0]
        if bucket_name == url_useful_part:
            prefix = ""
        else:
            prefix = url_useful_part.split(bucket_name + "/")[-1]

        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        file_urls = [blob.public_url for blob in blobs if blob.name.endswith(accepted_file_types)]
        return {"file_urls": file_urls}
    else:
        raise ValueError(f"Cloud provider '{cloud_provider}' is not supported.")


# @output_on_top
# @op("LynxScribe RAG Graph Vector Store")
# @mem.cache
# def ls_rag_graph(
#     *,
#     name: str = "faiss",
#     num_dimensions: int = 3072,
#     collection_name: str = "lynx",
#     text_embedder_interface: str = "openai",
#     text_embedder_model_name_or_path: str = "text-embedding-3-large",
#     # api_key_name: str = "OPENAI_API_KEY",
# ):
#     """
#     Returns with a vector store instance.
#     """

#     # getting the text embedder instance
#     llm_params = {"name": text_embedder_interface}
#     # if api_key_name:
#     #     llm_params["api_key"] = os.getenv(api_key_name)
#     llm = get_llm_engine(**llm_params)
#     text_embedder = TextEmbedder(llm=llm, model=text_embedder_model_name_or_path)

#     # getting the vector store
#     if name == "chromadb":
#         vector_store = get_vector_store(name=name, collection_name=collection_name)
#     elif name == "faiss":
#         vector_store = get_vector_store(name=name, num_dimensions=num_dimensions)
#     else:
#         raise ValueError(f"Vector store name '{name}' is not supported.")

#     # building up the RAG graph
#     rag_graph = RAGGraph(
#         PandasKnowledgeBaseGraph(vector_store=vector_store, text_embedder=text_embedder)
#     )

#     return {"rag_graph": rag_graph}


@op("LynxScribe Image Describer")
@mem.cache
async def ls_image_describer(
    file_urls,
    *,
    llm_interface: str = "openai",
    llm_visual_model: str = "gpt-4o",
    llm_prompt_path: str = "uploads/image_description_prompts.yaml",
    llm_prompt_name: str = "cot_picture_descriptor",
    # api_key_name: str = "OPENAI_API_KEY",
):
    """
    Returns with image descriptions from a list of image URLs.

    TODO: making the inputs more flexible (e.g. accepting file locations, URLs, binaries, etc.).
          the input dictionary should contain some meta info: e.g., what is in the list...
    """

    # handling inputs
    image_urls = file_urls["file_urls"]

    # loading the LLM
    llm_params = {"name": llm_interface}
    # if api_key_name:
    #     llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)

    # preparing the prompts
    prompt_base = load_config(llm_prompt_path)[llm_prompt_name]
    prompt_list = []

    for i in range(len(image_urls)):
        image = image_urls[i]

        _prompt = deepcopy(prompt_base)
        for message in _prompt:
            if isinstance(message["content"], list):
                for _message_part in message["content"]:
                    if "image_url" in _message_part:
                        _message_part["image_url"] = {"url": image}

        prompt_list.append(_prompt)

    # creating the prompt objects
    ch_prompt_list = [
        ChatCompletionPrompt(model=llm_visual_model, messages=prompt) for prompt in prompt_list
    ]

    # get the image descriptions
    tasks = [llm.acreate_completion(completion_prompt=_prompt) for _prompt in ch_prompt_list]
    out_completions = await asyncio.gather(*tasks)
    results = [
        dictionary_corrector(result.choices[0].message.content) for result in out_completions
    ]

    # getting the image descriptions (list of dictionaries {image_url: URL, description: description})
    # TODO: some result class could be a better idea (will be developed in LynxScribe)
    image_descriptions = [
        {"image_url": image_urls[i], "description": results[i]} for i in range(len(image_urls))
    ]

    return {"image_descriptions": image_descriptions}


@op("LynxScribe Image RAG Builder")
@mem.cache
async def ls_image_rag_builder(
    image_descriptions,
    *,
    vdb_provider_name: str = "faiss",
    vdb_num_dimensions: int = 3072,
    vdb_collection_name: str = "lynx",
    text_embedder_interface: str = "openai",
    text_embedder_model_name_or_path: str = "text-embedding-3-large",
    # api_key_name: str = "OPENAI_API_KEY",
):
    """
    Based on image descriptions, and embedding/VDB parameters,
    the function builds up an image RAG graph, where the nodes are the
    descriptions of the images (and of all image objects).

    In a later phase, synthetic questions and "named entities" will also
    be added to the graph.
    """

    # handling inputs
    image_descriptions = image_descriptions["image_descriptions"]

    # Building up the empty RAG graph

    # a) Define LLM interface and get a text embedder
    llm_params = {"name": text_embedder_interface}
    # if api_key_name:
    #     llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)
    text_embedder = TextEmbedder(llm=llm, model=text_embedder_model_name_or_path)

    # b) getting the vector store
    # TODO: vdb_provider_name should be ENUM, and other parameters should appear accordingly
    if vdb_provider_name == "chromadb":
        vector_store = get_vector_store(name=vdb_provider_name, collection_name=vdb_collection_name)
    elif vdb_provider_name == "faiss":
        vector_store = get_vector_store(name=vdb_provider_name, num_dimensions=vdb_num_dimensions)
    else:
        raise ValueError(f"Vector store name '{vdb_provider_name}' is not supported.")

    # c) building up the RAG graph
    rag_graph = RAGGraph(
        PandasKnowledgeBaseGraph(vector_store=vector_store, text_embedder=text_embedder)
    )

    dict_list_df = []
    for image_description_tuple in image_descriptions:
        image_url = image_description_tuple["image_url"]
        image_description = image_description_tuple["description"]

        if "overall description" in image_description:
            dict_list_df.append(
                {
                    "image_url": image_url,
                    "description": image_description["overall description"],
                    "source": "overall description",
                }
            )

        if "details" in image_description:
            for dkey in image_description["details"].keys():
                text = f"The picture's description is: {image_description['overall description']}\n\nThe description of the {dkey} is: {image_description['details'][dkey]}"
                dict_list_df.append(
                    {"image_url": image_url, "description": text, "source": "details"}
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

    return {"rag_graph": rag_graph}


@op("LynxScribe RAG Graph Saver")
def ls_save_rag_graph(
    rag_graph,
    *,
    image_rag_out_path: str = "image_test_rag_graph.pickle",
):
    """
    Saves the RAG graph to a pickle file.
    """

    # reading inputs
    rag_graph = rag_graph[0]["rag_graph"]

    rag_graph.kg_base.save(image_rag_out_path)
    return None


@ops.input_position(rag_graph="bottom")
@op("LynxScribe Image RAG Query")
async def search_context(rag_graph, text, *, top_k=3):
    """
    top_k: which results we are showing (TODO: when the image viewer is
    updated w pager, change back to top k)
    """
    message = text["text"]
    rag_graph = rag_graph[0]["rag_graph"]

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
        result_list.append({"image_url": image_url, "score": score, "description": description})

    real_k = min(top_k, len(result_list) - 1)

    return {"embedding_similarities": [result_list[real_k]]}


@op("LynxScribe Image Result Viewer", view="image")
def view_image(embedding_similarities):
    """
    Plotting the TOP images (from embedding similarities).

    TODO: later on, the user can scroll the images and send feedbacks
    """
    embedding_similarities = embedding_similarities["embedding_similarities"]
    return embedding_similarities[0]["image_url"]


@op("LynxScribe Text RAG Loader")
@mem.cache
def ls_text_rag_loader(
    file_urls,
    *,
    input_type: RAGVersion = RAGVersion.V1,
    vdb_provider_name: str = "faiss",
    vdb_num_dimensions: int = 3072,
    vdb_collection_name: str = "lynx",
    text_embedder_interface: str = "openai",
    text_embedder_model_name_or_path: str = "text-embedding-3-large",
    # api_key_name: str = "OPENAI_API_KEY",
):
    """
    Loading a text-based RAG graph from saved files (getting pandas readable links).
    """

    # handling inputs
    file_urls = file_urls["file_urls"]

    # getting the text embedder instance
    llm_params = {"name": text_embedder_interface}
    # if api_key_name:
    #     llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)
    text_embedder = TextEmbedder(llm=llm, model=text_embedder_model_name_or_path)

    # getting the vector store
    if vdb_provider_name == "chromadb":
        vector_store = get_vector_store(name=vdb_provider_name, collection_name=vdb_collection_name)
    elif vdb_provider_name == "faiss":
        vector_store = get_vector_store(name=vdb_provider_name, num_dimensions=vdb_num_dimensions)
    else:
        raise ValueError(f"Vector store name '{vdb_provider_name}' is not supported.")

    # building up the RAG graph
    rag_graph = RAGGraph(
        PandasKnowledgeBaseGraph(vector_store=vector_store, text_embedder=text_embedder)
    )

    # loading the knowledge base (temporary + TODO: adding v2)
    if input_type == RAGVersion.V1:
        node_file = [f for f in file_urls if "nodes.p" in f][0]
        edge_file = [f for f in file_urls if "edges.p" in f][0]
        tempcluster_file = [f for f in file_urls if "clusters.p" in f][0]
        rag_graph.kg_base.load_v1_knowledge_base(
            nodes_path=node_file,
            edges_path=edge_file,
            template_cluster_path=tempcluster_file,
        )
    elif input_type == RAGVersion.V2:
        raise ValueError("Currently only v1 input type is supported.")
    else:
        raise ValueError(f"Input type '{input_type}' is not supported.")

    return {"rag_graph": rag_graph}


@op("LynxScribe FAQ to RAG")
@mem.cache
async def ls_faq_to_rag(
    *,
    faq_excel_path: str = "",
    vdb_provider_name: str = "faiss",
    vdb_num_dimensions: int = 3072,
    vdb_collection_name: str = "lynx",
    text_embedder_interface: str = "openai",
    text_embedder_model_name_or_path: str = "text-embedding-3-large",
    scenario_cluster_distance_pct: int = 30,
):
    """
    Loading a text-based RAG graph from saved files (getting pandas readable links).
    """

    # getting the text embedder instance
    llm_params = {"name": text_embedder_interface}
    llm = get_llm_engine(**llm_params)
    text_embedder = TextEmbedder(llm=llm, model=text_embedder_model_name_or_path)

    # getting the vector store
    if vdb_provider_name == "chromadb":
        vector_store = get_vector_store(name=vdb_provider_name, collection_name=vdb_collection_name)
    elif vdb_provider_name == "faiss":
        vector_store = get_vector_store(name=vdb_provider_name, num_dimensions=vdb_num_dimensions)
    else:
        raise ValueError(f"Vector store name '{vdb_provider_name}' is not supported.")

    # building up the RAG graph
    rag_graph = RAGGraph(
        PandasKnowledgeBaseGraph(vector_store=vector_store, text_embedder=text_embedder)
    )

    # loading the knowledge base from the FAQ file
    rag_template = RAGTemplate.from_excel_path(
        path=faq_excel_path,
        faq_data_sheet_name="scenario_examples",
        scenario_data_sheet_name="scenario_scripts",
        prompt_codes_sheet_name="prompt_dictionary",
    )

    faq_loader_params = {
        "id_column": "scenario_example_ID",
        "timestamp_column": "last_modified_timestamp",
        "validity_column": "valid_flg",
        "question_type_contents_id": ["faq_question", "faq_question", "q_{id}"],
        "answer_type_contents_id": ["faq_answer", "{faq_question}\n\n{faq_answer}", "a_{id}"],
        "question_to_answer_edge_type_weight": ["qna", 1.0],
    }

    nodes, edges = FAQTemplateLoader(**faq_loader_params).load_nodes_and_edges(
        rag_template.faq_data
    )

    await rag_graph.kg_base.upsert_nodes(*nodes)
    rag_graph.kg_base.upsert_edges(edges)

    # Generating scenario clusters
    question_ids = [_id for _id in nodes[0] if _id.startswith("q_")]
    stored_embeddings = rag_graph.kg_base.vector_store.get(
        question_ids, include=["embeddings", "metadatas"]
    )
    embedding_vals = pd.Series([_emb.value for _emb in stored_embeddings], index=question_ids)
    labels = pd.Series(
        [_emb.metadata["scenario_name"] for _emb in stored_embeddings], index=question_ids
    )
    temp_cls = FclusterBasedClustering(distance_percentile=scenario_cluster_distance_pct)
    temp_cls.fit(embedding_vals, labels)
    df_tempclusters = temp_cls.get_cluster_centers()

    # Adding the scenario clusters to the RAG Graph
    df_tempclusters["template_id"] = "t_" + df_tempclusters.index.astype(str)
    df_tempclusters["embedding"] = df_tempclusters.apply(
        lambda row: Embedding(
            id=row["template_id"],
            value=row["cluster_center"],
            metadata={"scenario_name": row["control_label"], "type": "intent_cluster"},
        ),
        axis=1,
    )
    embedding_list = df_tempclusters["embedding"].tolist()
    rag_graph.kg_base.vector_store.upsert(embedding_list)

    return {"rag_graph": rag_graph}


@output_on_top
@op("LynxScribe RAG Graph Chatbot Builder")
def ls_rag_chatbot_builder(
    rag_graph,
    *,
    scenario_file: str = "uploads/lynx_chatbot_scenario_selector.yaml",
    node_types: str = "intent_cluster",
    scenario_meta_name: str = "",
):
    """
    Builds up a RAG Graph-based chatbot (basically the loaded RAG graph +
    a scenario selector).

    TODO: Later, the scenario selector can be built up synthetically from
    the input documents - or semi-automated, not just from the scenario
    yaml.
    """

    scenarios = load_config(scenario_file)
    node_types = [t.strip() for t in node_types.split(",")]

    # handling inputs
    rag_graph = rag_graph["rag_graph"]

    parameters = {
        "scenarios": [Scenario(**scenario) for scenario in scenarios],
        "node_types": node_types,
    }
    if len(scenario_meta_name) > 0:
        parameters["get_scenario_name"] = lambda node: node.metadata[scenario_meta_name]

    # loading the scenarios
    scenario_selector = ScenarioSelector(**parameters)

    # TODO: later we should unify this "knowledge base" object across the functions
    # this could be always an input of a RAG Chatbot, but also for other apps.
    return {
        "knowledge_base": {
            "rag_graph": rag_graph,
            "scenario_selector": scenario_selector,
        }
    }


@output_on_top
@ops.input_position(knowledge_base="bottom", chat_processor="bottom")
@op("LynxScribe RAG Graph Chatbot Backend")
def ls_rag_chatbot_backend(
    knowledge_base,
    chat_processor,
    *,
    negative_answer=DEFAULT_NEGATIVE_ANSWER,
    retriever_limits_by_type="{}",
    retriever_strict_limits=True,
    retriever_overall_chunk_limit=20,
    retriever_overall_token_limit=3000,
    retriever_max_iterations=3,
    llm_interface: str = "openai",
    llm_model_name: str = "gpt-4o",
    # api_key_name: str = "OPENAI_API_KEY",
):
    """
    Returns with a chatbot instance.
    """

    # handling_inputs
    rag_graph = knowledge_base[0]["knowledge_base"]["rag_graph"]
    scenario_selector = knowledge_base[0]["knowledge_base"]["scenario_selector"]
    chat_processor = chat_processor[0]["chat_processor"]
    limits_by_type = json.loads(retriever_limits_by_type)

    # connecting to the LLM
    llm_params = {"name": llm_interface}
    # if api_key_name:
    #     llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)

    # setting the parameters
    params = {
        "limits_by_type": limits_by_type,
        "strict_limits": retriever_strict_limits,
        "max_results": retriever_overall_chunk_limit,
        "token_limit": retriever_overall_token_limit,
        "max_iterations": retriever_max_iterations,
    }

    # generating the RAG Chatbot
    rag_chatbot = RAGChatbot(
        rag_graph=rag_graph,
        scenario_selector=scenario_selector,
        llm=llm,
        negative_answer=negative_answer,
        **params,
    )

    # generating the chatbot back-end
    c = ChatAPI(
        chatbot=rag_chatbot,
        chat_processor=chat_processor,
        model=llm_model_name,
    )

    return {"chat_api": c}


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
@op("LynxScribe Message")
def lynxscribe_message(
    *, prompt_role: MessageRole = MessageRole.SYSTEM, prompt_content: ops.LongStr
):
    return_message = Message(role=prompt_role.value, content=prompt_content.strip())
    return {"prompt_message": return_message}


@op("Read Excel")
def read_excel(*, file_path: str, sheet_name: str = "Sheet1", columns: str = ""):
    """
    Reads an Excel file and returns the content of the specified sheet.
    The columns parameter can be used to specify which columns to include in the output.
    If not specified, all columns will be included (separate the values by comma).

    TODO: more general: several input/output versions.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if columns:
        columns = [c.strip() for c in columns.split(",") if c.strip()]
        columns = [c for c in columns if c in df.columns]
        if len(columns) == 0:
            raise ValueError("No valid columns specified.")
        df = df[columns].copy()
    return {"dataframe": df}


@ops.input_position(system_prompt="bottom", instruction_prompt="bottom", dataframe="left")
@op("LynxScribe Task Solver")
@mem.cache
async def ls_task_solver(
    system_prompt,
    instruction_prompt,
    dataframe,
    *,
    llm_interface: str = "openai",
    llm_model_name: str = "gpt-4o",
    new_column_names: str = "processed_field",
    # api_key_name: str = "OPENAI_API_KEY",
):
    """
    Solving the described task on a data frame and put the results into a new column.

    If there are multiple new_column_names provided, the structured dictionary output
    will be split into multiple columns.
    """

    # handling inputs
    system_message = system_prompt[0]["prompt_message"]
    instruction_message = instruction_prompt[0]["prompt_message"]
    df = dataframe["dataframe"]

    # preparing output
    out_df = df.copy()

    # connecting to the LLM
    llm_params = {"name": llm_interface}
    # if api_key_name:
    #     llm_params["api_key"] = os.getenv(api_key_name)
    llm = get_llm_engine(**llm_params)

    # getting the list of fieldnames used in the instruction message
    fieldnames = []
    for pot_fieldname in df.columns:
        if "{" + pot_fieldname + "}" in instruction_message.content:
            fieldnames.append(pot_fieldname)

    # generate a list of instruction messages (from fieldnames)
    # each row of the df is a separate instruction message
    # TODO: make it fast for large dataframes
    instruction_messages = []
    for i in range(len(df)):
        instruction_message_i = deepcopy(instruction_message)
        for fieldname in fieldnames:
            instruction_message_i.content = instruction_message_i.content.replace(
                "{" + fieldname + "}", str(df.iloc[i][fieldname])
            )
        instruction_messages.append(instruction_message_i)

    # generate completition prompt
    completion_prompts = [
        ChatCompletionPrompt(
            model=llm_model_name,
            messages=[system_message, instruction_message_j],
        )
        for instruction_message_j in instruction_messages
    ]

    # get the answers
    tasks = [llm.acreate_completion(completion_prompt=_prompt) for _prompt in completion_prompts]
    out_completions = await asyncio.gather(*tasks)

    # answer post-processing: 1 vs more columns
    col_list = [_c.strip() for _c in new_column_names.split(",") if _c.strip()]
    if len(col_list) == 0:
        raise ValueError("No valid column names specified.")
    elif len(col_list) == 1:
        out_df[col_list[0]] = [result.choices[0].message.content for result in out_completions]
    else:
        answers = [
            dictionary_corrector(result.choices[0].message.content, expected_keys=col_list)
            for result in out_completions
        ]
        for i, col in enumerate(col_list):
            out_df[col] = [answer[col] for answer in answers]

    return {"dataframe": out_df}


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
    if len(response.choices) == 0:
        answer = "The following FAQ items are similar to the question:\n"
        for item in response.sources:
            answer += f"------------------------------------------------------ \n{item.body}\n\n"
    else:
        answer = response.choices[0].message.content
    if show_details:
        return {"answer": answer, **response.__dict__}
    else:
        return {"answer": answer}


@op("Input chat")
def input_chat(*, chat: str):
    return {"text": chat}


@ops.input_position(input="bottom")
@op("View DataFrame", view="table_view")
def view_df(input):
    df = input[0]["dataframe"]
    v = {
        "dataframes": {
            "df": {
                "columns": [str(c) for c in df.columns],
                "data": df.values.tolist(),
            }
        }
    }
    return v


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


async def get_chat_api(ws: str):
    from lynxkite.core import workspace

    cwd = pathlib.Path()
    path = cwd / ws
    assert path.is_relative_to(cwd)
    assert path.exists(), f"Workspace {path} does not exist"
    ws = workspace.load(path)
    contexts = await ops.EXECUTORS[ENV](ws)
    nodes = [op for op in ws.nodes if op.data.title == "LynxScribe RAG Graph Chatbot Backend"]
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
    from lynxkite.core import workspace

    workspaces = []
    for p in pathlib.Path().glob("**/*"):
        if p.is_file():
            try:
                ws = workspace.load(p)
                if ws.env == ENV:
                    workspaces.append(p)
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
        trf_dict = json.loads(dstring_prc)
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
