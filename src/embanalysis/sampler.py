import warnings
import numpy as np
import pandas as pd

from embanalysis.constants import DEFAULT_SEED
from embanalysis.extractor import HFEmbeddingsExtractor
from embanalysis.sample_data import (
    IntegerSampleMeta,
    RandomSampleMeta,
)
from embanalysis.tokenizer import HFTokenizerWrapper

from collections.abc import Iterable


def make_embeddings_df(
    token_ids: np.ndarray,
    tokens: Iterable[int | str],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token_id": token_ids,
            "token": tokens,
            "embeddings": [embeddings[i] for i in range(embeddings.shape[0])],
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

        df = make_embeddings_df(token_ids, tokens, embeddings)
        meta = IntegerSampleMeta(model_id=self.model_id)

        return df, meta

    def _random_token_ids(self, sample_size, seed):
        rng = np.random.default_rng(seed)
        return rng.choice(self.tokenizer.vocab_size, size=sample_size, replace=False)

    def random(
        self, sample_size=1000, seed=DEFAULT_SEED
    ) -> tuple[pd.DataFrame, RandomSampleMeta]:
        token_ids = self._random_token_ids(sample_size, seed)
        tokens = self.tokenizer.token_ids_to_tokens(token_ids)
        embeddings = self.extractor.extract(token_ids)

        df = make_embeddings_df(token_ids, tokens, embeddings)
        meta = RandomSampleMeta(
            model_id=self.model_id,
            sample_size=sample_size,
            seed=seed,
        )

        return df, meta
