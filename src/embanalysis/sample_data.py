from dataclasses import dataclass, field
import pandas as pd

from typing import Literal

from sklearn.base import BaseEstimator


type EmbeddingsSampleMeta = IntegerSampleMeta | RandomSampleMeta | ReducedSampleMeta


@dataclass
class IntegerSampleMeta:
    tag: Literal["integers"] = field(init=False, default="integers")
    model_id: str

    def label(self) -> str:
        return "Single Token Integers"


@dataclass
class RandomSampleMeta:
    tag: Literal["random"] = field(init=False, default="random")
    model_id: str
    sample_size: int
    seed: int

    def label(self) -> str:
        return f"Random sample (size={self.sample_size}, seed={self.seed})"


@dataclass
class ReducedSampleMeta:
    tag: Literal["reduced"] = field(init=False, default="reduced")
    original: EmbeddingsSampleMeta
    estimator: BaseEstimator

    @property
    def model_id(self) -> str:
        return self.original.model_id

    def label(self) -> str:
        return f"Reduced sample of {self.original.label()}"


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
