"""PyKEEN operations."""

from lynxkite_core import ops
from . import core
from .pytorch.pytorch_core import _torch_save, _torch_load

import typing
import pandas as pd
import enum
from pykeen import models
from pykeen import evaluation
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples, predict_target, predict_all
from pykeen.triples import TriplesFactory


op = ops.op_registration(core.ENV)


PyKEENModelName = typing.Annotated[
    str,
    {
        "format": "dropdown",
        "metadata_query": "[].other.*[] | [?type == 'pykeen-model'].key",
    },
]
"""A type annotation to be used for parameters of an operation. PyKEENModelName is
rendered as a dropdown in the frontend, listing the PyKEEN models in the Bundle.
The model name is passed to the operation as a string."""


def mapped_triples_to_df(triples, entity_to_id, relation_to_id):
    df = pd.DataFrame(triples.numpy(), columns=["head", "relation", "tail"])
    entity_label_mapping = {idx: label for label, idx in entity_to_id.items()}
    relation_label_mapping = {idx: label for label, idx in relation_to_id.items()}

    df["head"] = df["head"].map(entity_label_mapping)
    df["relation"] = df["relation"].map(relation_label_mapping)
    df["tail"] = df["tail"].map(entity_label_mapping)
    return df


class PyKEENDataset(str, enum.Enum):
    AristoV4 = "AristoV4"
    BioKG = "BioKG"
    CKG = "CKG"
    CN3l = "CN3l"
    CoDExLarge = "CoDExLarge"
    CoDExMedium = "CoDExMedium"
    CoDExSmall = "CoDExSmall"
    ConceptNet = "ConceptNet"
    Countries = "Countries"
    CSKG = "CSKG"
    DB100K = "DB100K"
    DBpedia50 = "DBpedia50"
    DRKG = "DRKG"
    FB15k = "FB15k"
    FB15k237 = "FB15k237"
    Globi = "Globi"
    Hetionet = "Hetionet"
    Kinships = "Kinships"
    Nations = "Nations"
    NationsLiteral = "NationsLiteral"
    OGBBioKG = "OGBBioKG"
    OGBWikiKG2 = "OGBWikiKG2"
    OpenBioLink = "OpenBioLink"
    OpenBioLinkLQ = "OpenBioLinkLQ"
    OpenEA = "OpenEA"
    PharMeBINet = "PharMeBINet"
    PharmKG = "PharmKG"
    PharmKG8k = "PharmKG8k"
    PrimeKG = "PrimeKG"
    UMLS = "UMLS"
    WD50KT = "WD50KT"
    Wikidata5M = "Wikidata5M"
    WK3l120k = "WK3l120k"
    WK3l15k = "WK3l15k"
    WN18 = "WN18"
    WN18RR = "WN18RR"
    YAGO310 = "YAGO310"

    def to_dataset(self):
        return get_dataset(dataset=self.value)


@op("Import PyKEEN dataset")
def import_pykeen_dataset_path(*, dataset: PyKEENDataset = PyKEENDataset.Nations) -> core.Bundle:
    ds = dataset.to_dataset()

    # Training -------------
    triples = ds.training.mapped_triples
    df_train = mapped_triples_to_df(
        triples=triples,
        entity_to_id=ds.entity_to_id,
        relation_to_id=ds.relation_to_id,
    )

    # Testing -------------
    triples = ds.testing.mapped_triples
    df_test = mapped_triples_to_df(
        triples=triples,
        entity_to_id=ds.entity_to_id,
        relation_to_id=ds.relation_to_id,
    )

    # Validation -----------
    triples = ds.validation.mapped_triples
    df_val = mapped_triples_to_df(
        triples=triples,
        entity_to_id=ds.entity_to_id,
        relation_to_id=ds.relation_to_id,
    )

    bundle = core.Bundle()
    bundle.dfs["edges_train"] = df_train
    bundle.dfs["edges_test"] = df_test
    bundle.dfs["edges_val"] = df_val

    bundle.dfs["nodes"] = pd.DataFrame(
        {
            "id": list(ds.entity_to_id.values()),
            "label": list(ds.entity_to_id.keys()),
        }
    )
    bundle.dfs["relations"] = pd.DataFrame(
        {
            "id": list(ds.relation_to_id.values()),
            "label": list(ds.relation_to_id.keys()),
        }
    )

    df_all = pd.concat([df_train, df_test, df_val], ignore_index=True)
    bundle.dfs["edges"] = pd.DataFrame(
        {
            "head": df_all["head"].tolist(),
            "tail": df_all["tail"].tolist(),
            "relation": df_all["relation"].tolist(),
        }
    )
    return bundle


class PyKEENModel(str, enum.Enum):
    AutoSF = "AutoSF"
    BoxE = "BoxE"
    Canonical_Tensor_Decomposition = "CP"
    CompGCN = "CompGCN"
    ComplEx = "ComplEx"
    ConvE = "ConvE"
    ConvKB = "ConvKB"
    Cooccurrence_Filtered = "CooccurrenceFilteredModel"
    CrossE = "CrossE"
    DistMA = "DistMA"
    DistMult = "DistMult"
    ER_MLP = "ERMLP"
    ER_MLPE = "ERMLPE"
    Fixed_Model = "FixedModel"
    HolE = "HolE"
    KG2E = "KG2E"
    MuRE = "MuRE"
    NTN = "NTN"
    NodePiece = "NodePiece"
    PairRE = "PairRE"
    ProjE = "ProjE"
    QuatE = "QuatE"
    RGCN = "RGCN"
    RESCAL = "RESCAL"
    RotatE = "RotatE"
    SimplE = "SimplE"
    Structured_Embedding = "SE"
    TorusE = "TorusE"
    TransD = "TransD"
    TransE = "TransE"
    TransF = "TransF"
    TransH = "TransH"
    TransR = "TransR"
    TuckER = "TuckER"

    def to_class(
        self, triples_factory: TriplesFactory, embedding_dim: int, seed: int = 42
    ) -> models.Model:
        return getattr(models, self.value)(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            random_seed=seed,
        )


class PyKEENModelWrapper:
    """Wrapper to add metadata method to PyKEEN models for dropdown queries, and to enable caching of model"""

    def __init__(
        self,
        model: models.Model,
        model_type: PyKEENModel,
        embedding_dim: int,
        entity_to_id: dict,
        relation_to_id: dict,
        edges_data: pd.DataFrame,
        seed: int,
        trained: bool = False,
    ):
        self.model = model
        self.trained = trained
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.edges_data = edges_data
        self.seed = seed

    def metadata(self) -> dict:
        return {
            "type": "pykeen-model",
            "model_class": self.model.__class__.__name__,
            "module": self.model.__module__,
            "trained": self.trained,
            "embedding_dim": self.embedding_dim,
        }

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped model
        # Use object.__getattribute__ to avoid recursion when accessing self.model
        model = object.__getattribute__(self, "model")
        return getattr(model, name)

    def __str__(self):
        return str(self.model)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["model"]
        if self.trained:
            state["model_state_dict"] = _torch_save(self.model.state_dict())

        return state

    def __setstate__(self, state: dict) -> None:
        model_state_dict = state.pop("model_state_dict", None)
        self.__dict__.update(state)
        if (
            self.model_type
            and self.embedding_dim
            and self.entity_to_id
            and self.relation_to_id
            and self.edges_data is not None
        ):
            self.model = self.model_type.to_class(
                triples_factory=TriplesFactory.from_labeled_triples(
                    self.edges_data.to_numpy(),
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    create_inverse_triples=req_inverse_triples(self.model_type),
                ),
                embedding_dim=self.embedding_dim,
                seed=self.seed,
            )
            if self.trained and model_state_dict is not None:
                self.model.load_state_dict(_torch_load(model_state_dict))

    def __repr__(self):
        return f"PyKEENModelWrapper({self.model.__class__.__name__})"


def req_inverse_triples(model: models.Model | PyKEENModel) -> bool:
    """
    Check if the model requires inverse triples.
    """
    return model in {
        models.CompGCN,
        models.NodePiece,
        PyKEENModel.CompGCN,
        PyKEENModel.NodePiece,
    }


class TrainingType(str, enum.Enum):
    sLCWA = "sLCWA"
    LCWA = "LCWA"

    def __str__(self):
        return self.value


@op("Define PyKEEN model")
def define_pykeen_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModel = PyKEENModel.TransE,
    embedding_dim: int = 50,
    seed: int = 42,
    save_as: str = "PyKEENmodel",
):
    """Defines a PyKEEN model based on the selected model type."""
    bundle = bundle.copy()
    entity_to_id = bundle.dfs["nodes"].set_index("label")["id"].to_dict()
    relation_to_id = bundle.dfs["relations"].set_index("label")["id"].to_dict()
    edges_data = bundle.dfs["edges"][["head", "relation", "tail"]]

    model_class = model.to_class(
        triples_factory=TriplesFactory.from_labeled_triples(
            edges_data.to_numpy(),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=req_inverse_triples(model),
        ),
        embedding_dim=embedding_dim,
        seed=seed,
    )
    model_wrapper = PyKEENModelWrapper(
        model_class,
        model_type=model,
        embedding_dim=embedding_dim,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        edges_data=edges_data,
        seed=seed,
    )
    bundle.other[save_as] = model_wrapper
    return bundle


class PyKEENSupportedOptimizers(str, enum.Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    Adamax = "Adamax"
    Adagrad = "Adagrad"
    SGD = "SGD"


# TODO: Make the pipeline more customizable, e.g. by allowing to pass additional parameters to the pipeline function.
@op("Train embedding model", slow=True)
def train_embedding_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModelName = "PyKEENmodel",
    training_table: core.TableName = "edges_train",
    testing_table: core.TableName = "edges_test",
    validation_table: core.TableName = "edges_val",
    optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam,
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
):
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model)
    actual_model = model_wrapper.model
    sampler = None
    if actual_model is models.RGCN and training_approach == TrainingType.sLCWA:
        # Currently RGCN is the only model that requires a sampler and only when using sLCWA
        sampler = "schlichtkrull"

    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id

    training_set = TriplesFactory.from_labeled_triples(
        bundle.dfs[training_table][["head", "relation", "tail"]].to_numpy(),
        create_inverse_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    testing_set = TriplesFactory.from_labeled_triples(
        bundle.dfs[testing_table][["head", "relation", "tail"]].to_numpy(),
        create_inverse_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    validation_set = TriplesFactory.from_labeled_triples(
        bundle.dfs[validation_table][["head", "relation", "tail"]].to_numpy(),
        create_inverse_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    result = pipeline(
        training=training_set,
        testing=testing_set,
        validation=validation_set,
        model=actual_model,
        epochs=epochs,
        training_loop=training_approach,
        training_kwargs=dict(
            sampler=sampler,
            continue_training=model_wrapper.trained,
        ),
        optimizer=optimizer_type,
        evaluator=None,
        random_seed=model_wrapper.seed,
        stopper="early",
        stopper_kwargs=dict(
            frequency=2,
            patience=15,
            relative_delta=0.0005,
            metric="hits@k",
        ),
    )

    model_wrapper.model = result.model
    model_wrapper.trained = True

    bundle.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    bundle.other[model] = model_wrapper

    return bundle


@op("Triples prediction")
def triple_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    table_name: core.TableName = "edges_val",
):
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)

    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id

    pred = predict_triples(
        model=model,
        triples=TriplesFactory.from_labeled_triples(
            bundle.dfs[table_name][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
    )
    df = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_test"][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )
    ).df
    bundle.dfs["pred"] = df
    return bundle


@op("Target prediction")
def target_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    head: str,
    relation: str,
    tail: str,
):
    """
    Leave the target prediction field empty
    """
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id
    pred = predict_target(
        model=model,
        head=head if head != "" else None,
        relation=relation if relation != "" else None,
        tail=tail if tail != "" else None,
        triples_factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_train"][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
    )

    pred_annotated = pred.add_membership_columns(
        validation=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_val"][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
        testing=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_test"][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle


@op("Full prediction", slow=True)
def full_predict(
    bundle: core.Bundle, *, model_name: PyKEENModelName = "PyKEENmodel", k: int | None = None
):
    """Pass k="" to keep all scores"""
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id
    pred = predict_all(model=model, k=k)
    pack = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_val"][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )
    )
    pred_annotated = pack.add_membership_columns(
        training=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_train"][["head", "relation", "tail"]].to_numpy(),
            create_inverse_triples=req_inverse_triples(model),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle


@op("Extract embeddings from PyKEEN model")
def extract_from_pykeen(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
):
    bundle = bundle.copy()
    model = bundle.other[model_name]

    actual_model = model
    while hasattr(actual_model, "base"):
        actual_model = actual_model.base

    entity_labels = list(bundle.dfs["nodes"]["label"])
    entity_embeddings = None
    if (
        hasattr(actual_model, "entity_representations")
        and len(actual_model.entity_representations) > 0
    ):
        entity_embeddings = actual_model.entity_representations[0]().detach().cpu()

    if entity_embeddings is None:
        raise AttributeError(
            f"Cannot extract entity embeddings from model type: {type(actual_model)}. "
            f"Available attributes: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}"
        )

    nodes_table = bundle.dfs["nodes"]
    if "embedding" not in nodes_table.columns:
        nodes_table["embedding"] = None
    nodes_table["embedding"] = pd.Series(
        [entity_embeddings[entity_labels.index(label)].numpy() for label in entity_labels]
    )
    bundle.dfs["nodes"] = nodes_table

    relation_labels = list(bundle.dfs["relations"]["label"])
    relation_embeddings = None
    if (
        hasattr(actual_model, "relation_representations")
        and len(actual_model.relation_representations) > 0
    ):
        relation_embeddings = actual_model.relation_representations[0]().detach().cpu()
    if relation_embeddings is None:
        raise AttributeError(
            f"Cannot extract relation embeddings from model type: {type(actual_model)}. "
            f"Available attributes: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}"
        )

    relations_table = bundle.dfs["relations"]
    if "embedding" not in relations_table.columns:
        relations_table["embedding"] = None
    relations_table["embedding"] = pd.Series(
        [relation_embeddings[relation_labels.index(label)].numpy() for label in relation_labels]
    )
    bundle.dfs["relations"] = relations_table

    return bundle


class EvaluatorTypes(str, enum.Enum):
    ClassificationEvaluator = "Classification Evaluator"
    MacroRankBasedEvaluator = "Macro Rank Based Evaluator"
    RankBasedEvaluator = "Rank Based Evaluator"

    def to_class(self) -> evaluation.Evaluator:
        return getattr(evaluation, self.name.replace(" ", ""))()


@op("Evaluate model")
def evaluate(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    evaluator_type: EvaluatorTypes = EvaluatorTypes.RankBasedEvaluator,
    metrics_str: str = "ALL",
):
    """Metrics are a comma separated list, "ALL" if all metrics are needed"""
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id
    evaluator = evaluator_type.to_class()
    testing_triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["edges_test"][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    training_triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["edges_train"][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    validation_triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["edges_val"][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    evaluated = evaluator.evaluate(
        model=model.model,
        mapped_triples=testing_triples.mapped_triples,
        additional_filter_triples=[
            training_triples.mapped_triples,
            validation_triples.mapped_triples,
        ],
        batch_size=32,
    )
    if metrics_str == "ALL":
        bundle.dfs["metrics"] = evaluated.to_df()
        return bundle

    metrics = metrics_str.split(",")
    metrics_df = pd.DataFrame(columns=["metric", "score"])

    for metric in metrics:
        metric = metric.strip()
        try:
            score = evaluated.get_metric(metric)
        except Exception as e:
            raise Exception(f"Possibly unknown metric: {metric}") from e
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([[metric, score]], columns=metrics_df.columns)]
        )

    bundle.dfs["metrics"] = metrics_df

    return bundle
