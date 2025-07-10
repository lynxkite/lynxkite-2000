"""PyKEEN operations."""

from lynxkite.core import ops
from . import core

import pandas as pd
import enum
import pykeen
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples, predict_target, predict_all
from pykeen.triples import TriplesFactory


op = ops.op_registration(core.ENV)


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


@op("Import PyKEEN Dataset")
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
    bundle.dfs["triples_train"] = df_train
    bundle.dfs["triples_test"] = df_test
    bundle.dfs["triples_val"] = df_val
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

    def to_class(self):
        return getattr(pykeen.models, self.value, None)


def req_inverse_triples(model: pykeen.models.Model) -> bool:
    """
    Check if the model requires inverse triples.
    """
    return model in {
        pykeen.models.CompGCN,
        pykeen.models.NodePiece,
    }


class TrainingType(str, enum.Enum):
    sLCWA = "sLCWA"
    LCWA = "LCWA"

    def __str__(self):
        return self.value


# TODO: Make the pipeline more customizable, e.g. by allowing to pass additional parameters to the pipeline function.
@op("Train Embedding Model", slow=True, cache=False)
def train_embedding_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModel = PyKEENModel.TransE,
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
):
    bundle = bundle.copy()
    if "model" in bundle.other:
        model = bundle.other["model"]
    else:
        sampler = None
        if model is PyKEENModel.RGCN and training_approach == TrainingType.sLCWA:
            # Currently RGCN is the only model that requires a sampler and only when using sLCWA
            sampler = "schlichtkrull"
        model = model.to_class()
    training_set = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_train"][["head", "relation", "tail"]].values,
        create_inverse_triples=req_inverse_triples(model),
    )
    testing_set = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_test"][["head", "relation", "tail"]].values,
        create_inverse_triples=req_inverse_triples(model),
    )

    result = pipeline(
        training=training_set,
        testing=testing_set,
        model=model,
        epochs=epochs,
        training_loop=training_approach,
        training_kwargs={"sampler": sampler},
        evaluator=None,
    )

    bundle.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    bundle.other["model"] = result.model

    return bundle


@op("Triples prediction")
def triple_predict(bundle: core.Bundle, *, table_name: str = "triples_val"):
    bundle = bundle.copy()
    model = bundle.other.get("model")
    pred = predict_triples(
        model=model,
        triples=TriplesFactory.from_labeled_triples(
            bundle.dfs[table_name][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        ),
    )
    df = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_test"][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        )
    ).df
    bundle.dfs["pred"] = df
    return bundle


@op("Target Prediction")
def target_predict(bundle: core.Bundle, *, head: str, relation: str, tail: str):
    """
    Leave the target prediction field empty
    """
    bundle = bundle.copy()
    model = bundle.other.get("model")
    pred = predict_target(
        model=model,
        head=head if head != "" else None,
        relation=relation if relation != "" else None,
        tail=tail if tail != "" else None,
        triples_factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_train"][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        ),
    )

    pred_annotated = pred.add_membership_columns(
        validation=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_val"][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        ),
        testing=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_test"][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        ),
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle


@op("Full prediction", slow=True)
def full_predict(bundle: core.Bundle, *, k: int = None):
    """Pass k=0 to keep all scores"""
    bundle = bundle.copy()
    model = bundle.other.get("model")
    pred = predict_all(model=model, k=k if k != 0 else None)
    pack = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_val"][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        )
    )
    pred_annotated = pack.add_membership_columns(
        training=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_train"][["head", "relation", "tail"]].values,
            create_inverse_triples=req_inverse_triples(model),
        )
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle


@op("Extract embeddings from PyKEEN model")
def extract_from_pykeen(bundle: core.Bundle, *, node_embedding_name: str, edge_embedding_name: str):
    bundle = bundle.copy()
    model = bundle.other["model"]
    triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_train"][["head", "relation", "tail"]].values,
        create_inverse_triples=req_inverse_triples(model),
    )

    actual_model = model
    while hasattr(actual_model, "base"):
        actual_model = actual_model.base

    entity_labels = list(triples.entity_to_id.keys())
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

    bundle.other[node_embedding_name] = entity_embeddings
    bundle.other["entity_to_index"] = {entity: idx for idx, entity in enumerate(entity_labels)}

    relation_labels = list(triples.relation_to_id.keys())
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

    bundle.other[edge_embedding_name] = relation_embeddings
    bundle.other["relation_to_index"] = {rel: idx for idx, rel in enumerate(relation_labels)}

    return bundle
