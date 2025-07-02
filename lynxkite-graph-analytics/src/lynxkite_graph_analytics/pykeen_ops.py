"""PyKEEN operations."""

from lynxkite.core import ops
from . import core

import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples, predict_target, predict_all
from pykeen.triples import TriplesFactory
from pykeen import datasets

op = ops.op_registration(core.ENV)


@op("Import Pykeen Dataset")
def import_pykeen_dataset_path(*, dataset: str) -> core.Bundle:
    ds = 0
    if dataset in datasets.__all__:
        ds = get_dataset(dataset=dataset)
    else:
        ds = TriplesFactory.from_path(dataset)

    # Training -------------
    triples = ds.training.mapped_triples

    df_train = pd.DataFrame(triples.numpy(), columns=["head", "relation", "tail"])

    entity_label_mapping = {idx: label for label, idx in ds.entity_to_id.items()}
    relation_label_mapping = {idx: label for label, idx in ds.relation_to_id.items()}

    df_train["head"] = df_train["head"].map(entity_label_mapping)
    df_train["relation"] = df_train["relation"].map(relation_label_mapping)
    df_train["tail"] = df_train["tail"].map(entity_label_mapping)

    # Testing -------------
    triples = ds.testing.mapped_triples

    df_test = pd.DataFrame(triples.numpy(), columns=["head", "relation", "tail"])

    entity_label_mapping = {idx: label for label, idx in ds.entity_to_id.items()}
    relation_label_mapping = {idx: label for label, idx in ds.relation_to_id.items()}

    df_test["head"] = df_test["head"].map(entity_label_mapping)
    df_test["relation"] = df_test["relation"].map(relation_label_mapping)
    df_test["tail"] = df_test["tail"].map(entity_label_mapping)

    # Validation -----------
    triples = ds.validation.mapped_triples
    df_val = pd.DataFrame(triples.numpy(), columns=["head", "relation", "tail"])

    entity_label_mapping = {idx: label for label, idx in ds.entity_to_id.items()}
    relation_label_mapping = {idx: label for label, idx in ds.relation_to_id.items()}

    df_val["head"] = df_val["head"].map(entity_label_mapping)
    df_val["relation"] = df_val["relation"].map(relation_label_mapping)
    df_val["tail"] = df_val["tail"].map(entity_label_mapping)

    df = pd.concat([df_train, df_test])
    bundle = core.Bundle(dfs={"df": df})
    bundle.dfs["df_val"] = df_val
    return bundle


@op("Train Embedding Model", slow=True, cache=False)
def train_embedding_model(bundle: core.Bundle, *, model: str, epochs: int = 5):
    bundle = bundle.copy()
    training_set = TriplesFactory.from_labeled_triples(
        bundle.dfs["df_train"][["head", "relation", "tail"]].values
    )
    testing_set = TriplesFactory.from_labeled_triples(
        bundle.dfs["df_test"][["head", "relation", "tail"]].values
    )

    result = pipeline(
        training=training_set,
        testing=testing_set,
        model=model,
        epochs=epochs,
    )
    """
    print(result.losses)
    losses = []
    for loss in result.losses:
        losses.append(loss)
    """
    bundle.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    bundle.other["model"] = result.model
    bundle.other["result_train"] = result.training.triples
    return bundle


@op("Triples prediction")
def tail_predict(bundle: core.Bundle):
    bundle = bundle.copy()
    pred = predict_triples(
        model=bundle.other["model"],
        triples=TriplesFactory.from_labeled_triples(
            bundle.dfs["df_val"][["head", "relation", "tail"]].values
        ),
    )
    df = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["df_test"][["head", "relation", "tail"]].values
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
            bundle.dfs["df_train"][["head", "relation", "tail"]].values
        ),
    )

    pred_annotated = pred.add_membership_columns(
        validation=TriplesFactory.from_labeled_triples(
            bundle.dfs["df_val"][["head", "relation", "tail"]].values
        ),
        testing=TriplesFactory.from_labeled_triples(
            bundle.dfs["df_test"][["head", "relation", "tail"]].values
        ),
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle


@op("Full prediction")
def full_predict(bundle: core.Bundle, *, k: int = 15):
    bundle = bundle.copy()
    pred = predict_all(model=bundle.other["model"], k=k)
    pack = pred.process(
        factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["df_val"][["head", "relation", "tail"]].values
        )
    )
    pred_annotated = pack.add_membership_columns(
        training=TriplesFactory.from_labeled_triples(
            bundle.dfs["df_val"][["head", "relation", "tail"]].values
        )
    )
    bundle.dfs["pred"] = pred_annotated.df

    return bundle
