"""PyKEEN operations."""

from lynxkite.core import ops
from . import core

import pandas as pd
import enum
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


# TODO: Make the pipeline more customizable, e.g. by allowing to pass additional parameters to the pipeline function.
@op("Train Embedding Model", slow=True, cache=False)
def train_embedding_model(bundle: core.Bundle, *, model: str, epochs: int = 5):
    bundle = bundle.copy()
    if "model" in bundle.other:
        model = bundle.other["model"]
    training_set = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_train"][["head", "relation", "tail"]].values
    )
    testing_set = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_test"][["head", "relation", "tail"]].values
    )

    result = pipeline(
        training=training_set,
        testing=testing_set,
        model=model,
        epochs=epochs,
    )

    bundle.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    bundle.other["model"] = result.model

    return bundle


@op("Triples prediction")
def triple_predict(bundle: core.Bundle, *, table_name: str = "triples_val"):
    bundle = bundle.copy()
    pred = predict_triples(
        model=bundle.other["model"],
        triples=TriplesFactory.from_labeled_triples(
            bundle.dfs[table_name][["head", "relation", "tail"]].values
        ),
    )
    df = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_test"][["head", "relation", "tail"]].values
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
    pred = predict_target(
        model=bundle.other["model"],
        head=head if head != "" else None,
        relation=relation if relation != "" else None,
        tail=tail if tail != "" else None,
        triples_factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_train"][["head", "relation", "tail"]].values
        ),
    )

    pred_annotated = pred.add_membership_columns(
        validation=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_val"][["head", "relation", "tail"]].values
        ),
        testing=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_test"][["head", "relation", "tail"]].values
        ),
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle


@op("Full prediction", slow=True)
def full_predict(bundle: core.Bundle, *, k: int = None):
    """Pass k=0 to keep all scores"""
    bundle = bundle.copy()
    pred = predict_all(model=bundle.other["model"], k=k if k != 0 else None)
    pack = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_val"][["head", "relation", "tail"]].values
        )
    )
    pred_annotated = pack.add_membership_columns(
        training=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_train"][["head", "relation", "tail"]].values
        )
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle
