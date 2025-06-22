import warnings
import numpy as np
import pandas as pd

from embanalysis.extractor import HFEmbeddingsExtractor
from embanalysis.tokenizer import HFTokenizerWrapper

from typing import Literal, TypedDict, Hashable, Iterable


class IntegerSampleMeta(TypedDict):
    tag: Literal["integers"]


class RandomSampleMeta(TypedDict):
    tag: Literal["random"]
    sample_size: int
    seed: Hashable


type EmbeddingsSampleMeta = IntegerSampleMeta | RandomSampleMeta


def make_embeddings_df(
    token_ids: np.ndarray, tokens: Iterable[str], embeddings: np.ndarray, model_id: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": model_id,
            "token_id": token_ids,
            "token": tokens,
            "embeddings": np.split(embeddings, embeddings.shape[0], 0),
        }
    )


class HFEmbeddingsSampler:
    def __init__(self, tokenizer: HFTokenizerWrapper, extractor: HFEmbeddingsExtractor):
        self.tokenizer = tokenizer
        self.extractor = extractor

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.tokenizer.name_or_path}')"

    @property
    def model_id(self):
        return self.tokenizer.name_or_path

    @classmethod
    def from_model(cls, model_id):
        tokenizer = HFTokenizerWrapper.from_pretrained(model_id)
        embeddings_extractor = HFEmbeddingsExtractor(model_id)
        return cls(tokenizer, embeddings_extractor)

    def _single_token_integer_ids(self, max_value=10_000) -> Iterable[int]:
        for num in range(max_value):
            token_ids = self.tokenizer.tokenize(str(num)).squeeze()
            if token_ids.ndim == 0:
                yield token_ids.item()
            else:
                return
        warnings.warn(
            f"All integers from 0 to max_value={max_value} are single token. "
            "There may be more single-token integers."
        )

    def single_token_integers(self) -> tuple[pd.DataFrame, IntegerSampleMeta]:
        token_ids = np.fromiter(self._single_token_integer_ids(), int)
        tokens = range(len(token_ids))
        embeddings = self.extractor.extract(token_ids)

        df = make_embeddings_df(token_ids, tokens, embeddings, self.model_id)
        meta = {"tag": "integers"}

        return df, meta

    def _random_token_ids(self, sample_size, seed):
        rng = np.random.default_rng(seed)
        return rng.integers(low=0, high=self.tokenizer.vocab_size, size=sample_size)

    def random(self, sample_size=1000, seed=1234) -> tuple[pd.DataFrame, RandomSampleMeta]:
        token_ids = self._random_token_ids(seed, sample_size)
        tokens = self.tokenizer.token_ids_to_tokens(token_ids)
        embeddings = self.extractor.extract(token_ids)

        df = make_embeddings_df(token_ids, tokens, embeddings, self.model_id)
        meta = { "tag": "random", "sample_size": sample_size, "seed": seed }

        return df, meta
