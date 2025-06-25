from dataclasses import dataclass
import pandas as pd

from typing import Literal

from sklearn.base import BaseEstimator

type EmbeddingsSampleMeta = IntegerSampleMeta | RandomSampleMeta | ReducedSampleMeta


@dataclass
class IntegerSampleMeta:
    model_id: str

    tag: Literal["integers"] = "integers"
    def label(self) -> str:
        return "Single Token Integers"


@dataclass
class RandomSampleMeta:
    model_id: str
    sample_size: int
    seed: int

    tag: Literal["random"] = "random"
    def label(self) -> str:
        return f"Random sample (size={self.sample_size}, seed={self.seed})"


@dataclass
class ReducedSampleMeta:
    original: EmbeddingsSampleMeta
    estimator: BaseEstimator

    tag: Literal["reduced"] = "reduced"

    @property
    def model_id(self) -> str:
        return self.original.model_id

    def label(self) -> str:
        return f"{self.original.label()}: {self.estimator.__class__.__name__}"


def make_meta_object(meta: dict) -> EmbeddingsSampleMeta:
    """Convert a dictionary meta to an EmbeddingsSampleMeta instance."""
    match meta["tag"]:
        case "integers":
            return IntegerSampleMeta(**meta)
        case "random":
            return RandomSampleMeta(**meta)

    raise ValueError(f"Unknown tag in meta: {meta['tag']}")


@dataclass
class EmbeddingsSample[M: EmbeddingsSampleMeta = EmbeddingsSampleMeta]:
    sample_id: int
    meta: M
    embeddings_df: pd.DataFrame

    @property
    def model_id(self) -> str:
        return self.meta.model_id