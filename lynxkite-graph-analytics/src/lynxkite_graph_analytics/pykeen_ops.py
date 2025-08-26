"""PyKEEN operations."""

from lynxkite_core import ops
from . import core

import typing
import pandas as pd
import io
import enum
import torch
from torch import nn
from pykeen import models, evaluation, stoppers
from pykeen.nn import Interaction, combination
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.datasets import get_dataset
from pykeen.predict import predict_triples, predict_target, predict_all
from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory, leakage
from pykeen.models import LiteralModel


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


def mapped_triples_to_df(
    triples: torch.Tensor,
    entity_to_id: typing.Mapping[str, int],
    relation_to_id: typing.Mapping[str, int],
) -> pd.DataFrame:
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
    bundle = core.Bundle()

    # Training -------------
    triples = ds.training.mapped_triples
    df_train = mapped_triples_to_df(
        triples=triples,
        entity_to_id=ds.entity_to_id,
        relation_to_id=ds.relation_to_id,
    )
    bundle.dfs["edges_train"] = df_train

    # Testing -------------
    triples = ds.testing.mapped_triples
    df_test = mapped_triples_to_df(
        triples=triples,
        entity_to_id=ds.entity_to_id,
        relation_to_id=ds.relation_to_id,
    )
    bundle.dfs["edges_test"] = df_test

    # Validation -----------
    if ds.validation:
        triples = ds.validation.mapped_triples
        df_val = mapped_triples_to_df(
            triples=triples,
            entity_to_id=ds.entity_to_id,
            relation_to_id=ds.relation_to_id,
        )
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


class PyKEENCombinations(str, enum.Enum):
    ComplexSeparated = "ComplexSeparated"
    # Concat = "Concat"
    # ConcatAggregation = "ConcatAggregation"
    ConcatProjection = "ConcatProjection"
    Gated = "Gated"

    def to_class(
        self, embedding_dim: int, literal_shape: int, **kwargs
    ) -> combination.Combination:  # ty: ignore[invalid-return-type]
        match self:
            case "ComplexSeparated":
                return combination.ComplexSeparatedCombination(
                    combination=combination.ConcatProjectionCombination,
                    combination_kwargs=dict(
                        input_dims=[embedding_dim, literal_shape],
                        output_dim=embedding_dim,
                        bias=True,
                        activation=nn.Tanh,
                    ),
                )
            # case "Concat":
            #     return combination.ConcatCombination(
            #         dim=int(kwargs.get("dim"))
            #     )
            # case "ConcatAggregation":
            #     return combination.ConcatAggregationCombination('sum', aggregation_kwargs=dict(index=torch.tensor([0,1,2,3,4,1,2,3,4])))
            case "ConcatProjection":
                return combination.ConcatProjectionCombination(
                    input_dims=[embedding_dim, literal_shape],
                    output_dim=embedding_dim,
                    bias=kwargs.get("bias", False),
                    dropout=float(kwargs.get("dropout", 0.0)),
                    activation=kwargs.get("activation", "ReLU"),
                )
            case "Gated":
                return combination.GatedCombination(
                    entity_dim=embedding_dim,
                    literal_dim=literal_shape,
                    input_dropout=float(kwargs.get("dropout", 0.0)),
                    gate_activation=kwargs.get("gate_activation", "Sigmoid"),
                    hidden_activation=kwargs.get("hidden_activation", "Tanh"),
                )


class PyKEENModelWrapper:
    """Wrapper to add metadata method to PyKEEN models for dropdown queries, and to enable caching of model"""

    def __init__(
        self,
        model: models.Model,
        embedding_dim: int,
        entity_to_id: dict,
        relation_to_id: dict,
        edges_data: pd.DataFrame,
        seed: int,
        model_type: typing.Optional[PyKEENModel] = None,
        interaction: typing.Optional[Interaction] = None,
        inductive_inference: typing.Optional[pd.DataFrame] = None,
        inductive_kwargs: typing.Optional[dict] = None,
        combination: typing.Optional[PyKEENCombinations] = None,
        combination_kwargs: typing.Optional[dict] = None,
        literals_data: typing.Optional[pd.DataFrame] = None,
        trained: bool = False,
    ):
        if model_type is None:
            # either inductive or literals model
            if inductive_inference is not None:
                assert (
                    interaction is not None
                    and inductive_inference is not None
                    and combination is None
                    and literals_data is None
                ), (
                    "For inductive models, interaction and inductive inference table must be provided"
                )
            else:
                assert (
                    interaction is not None
                    and combination is not None
                    and literals_data is not None
                ), (
                    "For a model with literals, interaction, combination and literals data must be provided"
                )
        else:
            assert (
                interaction is None
                and combination is None
                and literals_data is None
                and inductive_inference is None
            ), "For transdutive models, only model_type must be provided"
        self.model = model
        self.embedding_dim = embedding_dim
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.edges_data = edges_data
        self.seed = seed
        self.model_type = model_type
        self.interaction = interaction
        self.combination = combination
        self.combination_kwargs = combination_kwargs
        self.literals_data = literals_data
        self.inductive_inference = inductive_inference
        self.inductive_kwargs = inductive_kwargs
        self.trained = trained

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
            buffer = io.BytesIO()
            self.model.save_state(buffer)
            state["model_state"] = buffer.getvalue()

        return state

    def __setstate__(self, state: dict) -> None:
        model_state = state.pop("model_state", None)
        self.__dict__.update(state)
        if self.model_type is not None:
            self.model = self.model_type.to_class(
                triples_factory=prepare_triples(
                    self.edges_data,
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    inv_triples=req_inverse_triples(self.model_type),
                ),
                embedding_dim=self.embedding_dim,
                seed=self.seed,
            )
        elif self.inductive_inference is not None:
            print(self.inductive_kwargs)
            model_cls = (
                models.InductiveNodePieceGNN
                if self.inductive_kwargs.get("use_GNN", False) == "True"
                else models.InductiveNodePiece
            )
            kwargs = self.inductive_kwargs
            kwargs.pop("use_GNN", None)
            self.model = model_cls(
                triples_factory=TriplesFactory.from_labeled_triples(
                    self.edges_data.to_numpy(),
                    relation_to_id=self.relation_to_id,
                    create_inverse_triples=True,
                ),
                inference_factory=prepare_triples(
                    self.inductive_inference.to_numpy(),
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    inv_triples=True,
                ),
                interaction=self.interaction,
                embedding_dim=self.embedding_dim,
                **kwargs,
            )
        else:
            combination_cls = self.combination.to_class(
                embedding_dim=self.embedding_dim,
                literal_shape=self.literals_data.shape[1],
                **(self.combination_kwargs or {}),
            )
            self.model = models.LiteralModel(
                triples_factory=prepare_triples(  # type: ignore[invalid-argument-type]
                    self.edges_data,
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    numeric_literals=self.literals_data,
                ),
                entity_representations_kwargs=dict(
                    shape=self.embedding_dim,
                ),
                relation_representations_kwargs=dict(
                    shape=self.embedding_dim,
                ),
                interaction=self.interaction,
                combination=combination_cls,
                random_seed=self.seed,
            )

        if self.trained and model_state is not None:
            buffer = io.BytesIO(model_state)
            self.model.load_state(buffer)

    def __repr__(self):
        return f"PyKEENModelWrapper({self.model.__class__.__name__})"


def req_inverse_triples(model: models.Model | PyKEENModel) -> bool:
    """
    Check if the model requires inverse triples.
    """
    return isinstance(
        model,
        (models.CompGCN, models.NodePiece, models.InductiveNodePiece, models.InductiveNodePieceGNN),
    ) or model in {PyKEENModel.CompGCN, PyKEENModel.NodePiece}


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
    edge_data_table: core.TableName = "edges",
    embedding_dim: int = 50,
    seed: int = 42,
    save_as: str = "PyKEENmodel",
):
    """Defines a PyKEEN model based on the selected model type."""
    bundle = bundle.copy()
    edges_data = bundle.dfs[edge_data_table][["head", "relation", "tail"]]
    edges_data["head"] = edges_data["head"].astype(str)
    edges_data["tail"] = edges_data["tail"].astype(str)
    triples_factory = TriplesFactory.from_labeled_triples(
        edges_data.to_numpy(),
        create_inverse_triples=req_inverse_triples(model),
    )

    model_class = model.to_class(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        seed=seed,
    )
    model_wrapper = PyKEENModelWrapper(
        model_class,
        model_type=model,
        embedding_dim=embedding_dim,
        entity_to_id=triples_factory.entity_to_id,
        relation_to_id=triples_factory.relation_to_id,
        edges_data=edges_data,
        seed=seed,
    )
    bundle.other[save_as] = model_wrapper
    return bundle


@op(
    "Define PyKEEN model with node attributes",
    params=[
        ops.ParameterGroup(
            name="combination_group",
            selector=ops.Parameter(
                name="combination_name",
                type=PyKEENCombinations,
                default=PyKEENCombinations.ConcatProjection,
            ),
            groups={
                "ComplexSeparated": [],
                # "Concat": [
                #     ops.Parameter.basic(name="dim", type=int, default=-1),
                # ],
                # "ConcatAggregation": [],
                "ConcatProjection": [
                    ops.Parameter.basic(name="bias", type=bool, default=False),
                    ops.Parameter.basic(name="dropout", type=float, default=0.0),
                    ops.Parameter.basic(name="activation", type=str, default="ReLU"),
                ],
                "Gated": [
                    ops.Parameter.basic(name="input_dropout", type=float, default=0.0),
                    ops.Parameter.basic(name="gate_activation", type=str, default="Sigmoid"),
                    ops.Parameter.basic(name="hidden_activation", type=str, default="Tanh"),
                ],
            },
            default=PyKEENCombinations.ConcatProjection,
        )
    ],
)
def def_pykeen_with_attributes(
    dataset: core.Bundle,
    *,
    interaction_name: PyKEENModel = PyKEENModel.TransE,
    combination_name: PyKEENCombinations = PyKEENCombinations.ConcatProjection,
    embedding_dim: int,
    random_seed: int,
    save_as: str,
    **kwargs,
) -> core.Bundle:
    dataset = dataset.copy()

    edges_data = dataset.dfs["edges"][["head", "relation", "tail"]].astype(str)
    triples_no_literals = TriplesFactory.from_labeled_triples(
        triples=edges_data.to_numpy(),
    )
    temp_model = interaction_name.to_class(
        triples_factory=triples_no_literals,
        embedding_dim=embedding_dim,
        seed=random_seed,
    )

    num_literals = dataset.dfs["literals"]
    if "node_id" not in num_literals.columns:
        raise ValueError("Expected a 'node_id' column in literals DataFrame.")
    num_literals["node_id"] = num_literals["node_id"].astype(str)
    order = [
        label for label, _ in sorted(triples_no_literals.entity_to_id.items(), key=lambda kv: kv[1])
    ]
    num_literals = num_literals.set_index("node_id").reindex(order)

    if num_literals.isna().any().any():
        raise ValueError("Some entities are missing literals after reindexing.")

    features = num_literals.reset_index(drop=True)

    dataset.dfs["literals"] = num_literals
    literals_to_id = {label: i for i, label in enumerate(features.columns)}

    combination_cls = combination_name.to_class(embedding_dim, len(features.columns), **kwargs)

    assert isinstance(temp_model, models.ERModel), "Only models derived from ERModel are supported."
    try:
        interaction: Interaction = temp_model.interaction
        entity_representations = temp_model.entity_representations
        relation_representations = temp_model.relation_representations
        # print(
        #     tuple(upgrade_to_sequence([entity_representation._embeddings for entity_representation in entity_representations])) + (Embedding,)
        # )
    except AttributeError as e:
        raise Exception(
            "Interaction not supported for this model type. Please use a different interaction."
        ) from e

    model = LiteralModel(
        triples_factory=TriplesNumericLiteralsFactory(
            mapped_triples=triples_no_literals.mapped_triples,
            entity_to_id=triples_no_literals.entity_to_id,
            relation_to_id=triples_no_literals.relation_to_id,
            numeric_literals=torch.from_numpy(features.to_numpy())
            .contiguous()
            .detach()
            .cpu()
            .numpy(),
            literals_to_id=literals_to_id,
        ),
        entity_representations=[
            entity_representation for entity_representation in entity_representations
        ],
        relation_representations=[
            relation_representation for relation_representation in relation_representations
        ],
        interaction=interaction,
        combination=combination_cls,
        random_seed=random_seed,
    )

    model_wrapper = PyKEENModelWrapper(
        model=model,
        interaction=model.interaction,
        combination=combination_name,
        combination_kwargs=kwargs,
        literals_data=features,
        embedding_dim=embedding_dim,
        entity_to_id=triples_no_literals.entity_to_id,
        relation_to_id=triples_no_literals.relation_to_id,
        edges_data=edges_data,
        seed=random_seed,
    )

    dataset.other[save_as] = model_wrapper
    return dataset


class PyKEENSupportedOptimizers(str, enum.Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    Adamax = "Adamax"
    Adagrad = "Adagrad"
    SGD = "SGD"


def prepare_triples(
    triples_df: pd.DataFrame,
    entity_to_id: dict,
    relation_to_id: dict,
    inv_triples: bool = False,
    numeric_literals: typing.Optional[pd.DataFrame] = None,
) -> TriplesFactory | TriplesNumericLiteralsFactory:
    """Prepare triples for PyKEEN from a DataFrame."""
    triples = TriplesFactory.from_labeled_triples(
        triples_df.astype(str).to_numpy(dtype=str),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=inv_triples,
    )
    if numeric_literals is not None:
        return TriplesNumericLiteralsFactory(
            mapped_triples=triples.mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            numeric_literals=torch.from_numpy(numeric_literals.to_numpy())
            .contiguous()
            .detach()
            .cpu()
            .numpy(),
            literals_to_id={label: i for i, label in enumerate(numeric_literals.columns)},
        )
    return triples


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
    loss_function: str = "BCEWithLogitsLoss",
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
):
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model)
    actual_model = model_wrapper.model
    sampler = None
    if isinstance(actual_model, models.RGCN) and training_approach == TrainingType.sLCWA:
        # Currently RGCN is the only model that requires a sampler and only when using sLCWA
        sampler = "schlichtkrull"

    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id

    training_set = prepare_triples(
        bundle.dfs[training_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    testing_set = prepare_triples(
        bundle.dfs[testing_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    validation_set = prepare_triples(
        bundle.dfs[validation_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    training_set, testing_set, validation_set = leakage.unleak(
        training_set, testing_set, validation_set
    )

    result: PipelineResult = pipeline(
        training=training_set,
        testing=testing_set,
        validation=validation_set,
        model=actual_model,
        loss=loss_function,
        optimizer=optimizer_type,
        optimizer_kwargs=dict(
            lr=0.00005,
        ),
        training_loop=training_approach,
        negative_sampler="bernoulli" if training_approach == TrainingType.sLCWA else None,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=64,
        ),
        epochs=epochs,
        training_kwargs=dict(
            sampler=sampler,
            continue_training=model_wrapper.trained,
        ),
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,
            patience=40,
            relative_delta=0.0005,
            metric="ah@k",
        ),
        random_seed=model_wrapper.seed,
    )

    model_wrapper.model = result.model
    model_wrapper.trained = True

    bundle.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    if isinstance(result.stopper, stoppers.EarlyStopper):
        bundle.dfs["early_stopper_metric"] = pd.DataFrame(
            {"early_stopper_metric": result.stopper.results}
        )
    bundle.other[model] = model_wrapper

    return bundle


@op("View early stopping metric", view="visualization")
def view_early_stopping(bundle: core.Bundle):
    metric = bundle.dfs["early_stopper_metric"].early_stopper_metric.tolist()
    v = {
        "title": {"text": "Early Stopping Metric"},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"data": metric, "type": "line"}],
    }
    return v


@op("Triples prediction")
def triple_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    table_name: core.TableName = "edges_val",
    inductive_setting: bool = False,
):
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)
    actual_model = model.model
    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id
    triples_to_predict_tf = prepare_triples(
        bundle.dfs[table_name][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        inv_triples=req_inverse_triples(actual_model) or inductive_setting,
    )

    if inductive_setting and isinstance(actual_model, models.InductiveERModel):
        original_repr = actual_model._get_entity_representations_from_inductive_mode(
            mode="validation"
        )
        actual_model.replace_entity_representations_(
            mode="validation",
            representation=actual_model.create_entity_representation_for_new_triples(
                triples_to_predict_tf
            ),
        )

    pred_df = (
        predict_triples(
            model=model,
            triples_factory=triples_to_predict_tf,
            mode="validation" if inductive_setting else None,
        )
        .process(
            factory=TriplesFactory(
                [[0, 0, 0]],  # Dummy triple to create a factory, as it is only used for mapping
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )
        )
        .df[["head_label", "relation_label", "tail_label", "score"]]
    )
    bundle.dfs["pred"] = pred_df
    if inductive_setting and isinstance(actual_model, models.InductiveERModel):
        # Restore the original entity representations after prediction
        actual_model.replace_entity_representations_(
            mode="validation", representation=original_repr
        )
    return bundle


@op("Target prediction")
def target_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    head: str,
    relation: str,
    tail: str,
    inductive_setting: bool = False,
):
    """
    Leave the target prediction field empty
    """
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    pred = predict_target(
        model=model_wrapper,
        head=head if head != "" else None,
        relation=relation if relation != "" else None,
        tail=tail if tail != "" else None,
        triples_factory=TriplesFactory(
            [[0, 0, 0]],  # Dummy triple to create a factory, as it is only used for mapping
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
        mode="validation" if inductive_setting else None,
    )

    col = "head_label" if head == "" else "tail_label" if tail == "" else "relation_label"
    df = pred.df[[col, "score"]]

    bundle.dfs["pred"] = df
    bundle.dfs["pred"].sort_values(by="score", ascending=False, inplace=True)
    return bundle


@op("Full prediction", slow=True)
def full_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    k: int | None = None,
    inductive_setting: bool = False,
):
    """
    Warning: This prediction can be a very expensive operation!

    Args:
        k: Pass "" to keep all scores
    """
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    pred = predict_all(model=model_wrapper, k=k, mode="validation" if inductive_setting else None)
    pack = pred.process(
        factory=TriplesFactory(
            [[0, 0, 0]],  # Dummy triple to create a factory, as it is only used for mapping
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
    )
    bundle.dfs["pred"] = pack.df[
        ["head_label", "relation_label", "tail_label", "score"]
    ].sort_values(by="score", ascending=False)

    return bundle


@op("Extract embeddings from PyKEEN model")
def extract_from_pykeen(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
):
    bundle = bundle.copy()
    model_wrapper = bundle.other[model_name]
    model = model_wrapper.model
    state_dict = model.state_dict()

    entity_embeddings = []
    for key in state_dict.keys():
        if "entity" in key.lower() and "embedding" in key.lower():
            entity_embeddings.append(state_dict[key].cpu().detach().numpy())

    id_to_entity = {v: k for k, v in model_wrapper.entity_to_id.items()}
    for i, embedding in enumerate(entity_embeddings):
        entity_embedding_df = pd.DataFrame({"embedding": list(embedding)})
        entity_embedding_df["node_label"] = entity_embedding_df.index.map(id_to_entity)
        bundle.dfs[f"node_embedding_{i}"] = entity_embedding_df

    relation_embeddings = []
    for key in state_dict.keys():
        if "relation" in key.lower() and "embedding" in key.lower():
            relation_embeddings.append(state_dict[key].cpu().detach().numpy())

    id_to_relation = {v: k for k, v in model_wrapper.relation_to_id.items()}
    for i, embedding in enumerate(relation_embeddings):
        relation_embedding_df = pd.DataFrame({"embedding": list(embedding)})
        relation_embedding_df["relation_label"] = relation_embedding_df.index.map(id_to_relation)
        bundle.dfs[f"relation_embedding_{i}"] = relation_embedding_df

    return bundle


class EvaluatorTypes(str, enum.Enum):
    ClassificationEvaluator = "Classification Evaluator"
    MacroRankBasedEvaluator = "Macro Rank Based Evaluator"
    RankBasedEvaluator = "Rank Based Evaluator"
    SampledRankBasedEvaluator = "Sampled Rank Based Evaluator"

    def to_class(self) -> evaluation.Evaluator:
        return getattr(evaluation, self.name.replace(" ", ""))()


@op("Evaluate model", slow=True)
def evaluate(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    evaluator_type: EvaluatorTypes = EvaluatorTypes.RankBasedEvaluator,
    eval_table: core.TableName = "edges_test",
    additional_true_triples_table: core.TableName = "edges_train",
    metrics_str: str = "ALL",
):
    """
    Evaluates the given model on the test set using the specified evaluator type.
    Args:
        evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
        metrics_str: Comma separated list, "ALL" if all metrics are needed.
    """

    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    evaluator = evaluator_type.to_class()
    if isinstance(evaluator, evaluation.ClassificationEvaluator):
        from pykeen.metrics.classification import classification_metric_resolver

        evaluator.metrics = tuple(
            classification_metric_resolver.make(metric_cls) for metric_cls in metrics_str.split(",")
        )
    testing_triples = prepare_triples(
        bundle.dfs[eval_table][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    additional_filters = prepare_triples(
        bundle.dfs[additional_true_triples_table][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    evaluated = evaluator.evaluate(
        model=model_wrapper.model,
        mapped_triples=testing_triples.mapped_triples,
        additional_filter_triples=additional_filters.mapped_triples,
        # batch_size=32,
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
