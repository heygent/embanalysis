# Implementation

The aim of the project was the realization of a framework to enable the analysis
and visualization of embeddings. To achieve this, we used three parts:

- an extraction and sampling layer, used to load models from HuggingFace and extracting
  the parts of the embeddings layer of interest for this inquiry.

- a storage layer, meant to store the selected samples in an easy to retrieve manner. In
  particular, during the course of this project we focused on sampling integers and some
  random embeddings, to have a comparison to see whether the structures formed were
  artifacts of the dimensionality reduction technique used.

- a CLI, to provide a user interface for the download, extraction and sampling of
  embeddings from HuggingFace models

- an analyzer, meant to compute base statistics about the embeddings and to extract
  results from the PCA analysis and give insights on various properties of the
  embeddings, such as explained variance of the various dimensions (through PCA) and
  correlation between numerical sequences and features.

- a visualizer, to create plots that give a visual intuition of the structures
  underneath the embedding data.

- a dashboard to display interactive visualization and to provide interactive data
  analysis features.

The following libraries have been employed in the making of this project:

- `typer` for implementing the CLI
- `transformers` and `torch` for model download and embeddings extraction
- `numpy`, `pandas`, `sklearn`, `sympy` and `umap` for math and calculation purposes
- `altair` and `plotly` for 2D and 3D plotting respectively
- `marimo` for notebook and reactive dashboard functionality

\clearpage

## Storage layer

The storage layer allows storing embeddings samples from any HuggingFace model without
loading the whole model, as doing so was very often impractically slow as a lot of the
work was done in a resource-constrained environment.

The sample data was stored in instances of the `EmbeddingsSample` class
([@lst:embeddingsample]), along with metadata reporting the source model ID and, for
random samples, the seed used for replicability purposes.

```python

@dataclass
class IntegerSampleMeta:
    model_id: str
    tag: Literal["integers"] = "integers"

@dataclass
class RandomSampleMeta:
model_id: str
    sample_size: int
    seed: int
    tag: Literal["random"] = "random"

@dataclass
class ReducedSampleMeta:
    original: EmbeddingsSampleMeta
    estimator: BaseEstimator
    tag: Literal["reduced"] = "reduced"

@dataclass
class EmbeddingsSample[M: EmbeddingsSampleMeta]:
    sample_id: int
    meta: M
    embeddings_df: pd.DataFrame = field(repr=False)
```

: Container classes for embeddings samples and their metadata. {#lst:embeddingsample}

Initially, the storage of each sample was done in a dedicated Parquet file, an efficient
file format that would have provided easy serialization of Pandas dataframes,
which were the main data structure employed in the analysis.
While initially adequate, this implementation didn't allow for easy sample
metadata storage, and required an ad-hoc cataloguing system based on filesystem
names to store and retrieve items on the basis of their metadata.

To address this, a choice was made to implement a more proper storage layer. It
was realized using DuckDB, a single-file database similar to SQLite that provides vector
functionality appropriate for the storage of embeddings. DuckDB also offers
facilities to work directly on dataframes using SQL queries, and exchanging data between
dataframes and the database in this way, which revealed very useful for loading
purposes.

```sql
CREATE OR REPLACE TABLE embeddings (
    model_id VARCHAR NOT NULL,
    token_id INTEGER NOT NULL,
    token VARCHAR NOT NULL,
    embeddings FLOAT[] NOT NULL,
    PRIMARY KEY (model_id, token_id),
);

CREATE OR REPLACE SEQUENCE embedding_id_seq;
CREATE OR REPLACE TABLE samples (
    sample_id INTEGER DEFAULT NEXTVAL('embedding_id_seq') PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    meta JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);

CREATE OR REPLACE TABLE embedding_to_sample (
    model_id VARCHAR NOT NULL,
    token_id INTEGER NOT NULL,
    sample_id INTEGER NOT NULL,
    PRIMARY KEY (model_id, token_id, sample_id),
);
```

: SQL schema for the storage layer. {#lst:sqlschema}

\clearpage

## Extraction and Sampling

Extraction is performed by downloading models using HuggingFace's Transformers
library, which allows for download and deployment of popular open source models.

```python
class HFEmbeddingsExtractor:
    """Extracts embeddings from a Hugging Face model."""

    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path

    @cached_property
    def embeddings(self):
        model = AutoModel.from_pretrained(self.name_or_path)
        model.eval()
        embeddings = model.embed_tokens
        return embeddings

    def extract(self, token_ids):
        with torch.no_grad():
            token_ids = torch.tensor(token_ids)
            return self.embeddings.forward(token_ids).squeeze().numpy()
```

: Extraction class for embeddings {#lst:extractor}

LLMs that make use of a tokenization step receive their sentences in input as a
list of token IDs, where each token ID corresponds to an embedding vector. It is
the LLM's tokenizer responsibility to take sentences, split them at the
appropriate token boundary, adding special tokens where necessary, and convert
them into token IDs for the LLM processing.

The logic to do this is split between the `HFTokenizerWrapper` ([@lst:tokenizer]) and
`HFEmbeddingsSampler` ([@lst:sampler]) classes. `HFTokenizerWrapper` invokes the
tokenizer to get the token IDs that correspond to the embeddings of interest (avoiding
special tokens, like `<Beginning of Sentence>` and such), while `HFEmbeddingsSampler`
has the logic for integer and random selection.

```python
class HFTokenizerWrapper:
    """Wrapper for Hugging Face tokenizers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, tokens) -> torch.Tensor:
        return self.tokenizer(
            tokens,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]

    @classmethod
    def from_pretrained(cls, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return cls(tokenizer)

    def token_ids_to_tokens(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return [token if token is not None else "<unk>" for token in tokens]
```

: HFTokenizerWrapper class, providing utility functions for tokenization.
{#lst:tokenizer}

The sampling happens by first picking the tokens of interest. For
numbers, we first verify that the model uses a tokenization scheme useful for
the numeric analysis intended, by trying to tokenize each integer from 0 onwards,
up to a maximum ceiling of 10.000, until we find the first integer that gets tokenized
using more than one token. The analysis is limited in scope to single-token integers in
the range 0-999, as these parameters correspond to a multitude of open source models
available as of today. For random sampling, token IDs are picked by doing a random
extraction of numbers between 0 and the vocabulary size of the model, and then
proceeding similarly.

After the sampling process is completed, the results are returned as a dataframe, along
with the corresponding provenance metadata.

```python
class HFEmbeddingsSampler:
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

    ...
    def _random_token_ids(self, sample_size, seed):
        rng = np.random.default_rng(seed)
        return rng.choice(self.tokenizer.vocab_size, size=sample_size, replace=False)
    ...
```

: Code for `HFEmbeddingSampler` {#lst:sampler}


\clearpage

## Analysis

Most of the analysis in done in the `EmbeddingsAnalyzer` ([@lst:embeddingsanalyzer])
class, which reorganizes data and provides it in a format suitable for consultation and
visualization.

```python
@dataclass
class EmbeddingsAnalyzer:
    embeddings_df: pd.DataFrame
    meta: EmbeddingsSampleMeta

    @classmethod
    def from_sample(cls, sample: EmbeddingsSample):
        """Initialize from an EmbeddingsSample."""
        return cls(
            embeddings_df=wide_embeddings_df(sample.embeddings_df),
            meta=sample.meta,
        )
```

: Initializing code for `EmbeddingsAnalyzer` {#lst:embeddingsanalyzer}

The format used for the embeddings here is a dataframe with the columns `token`,
`token_id` and the embeddings spread out in columns named `embeddings_{dimension
index}`. Even though it's a little unwieldy, this allows for compatibility with most
of the libraries operating with dataframe that assume mono-dimensional column indices
with string column names. This class also provides the facility for dimensional
reduction through the `run_estimator` method, which takes an estimator as input and
returns a new `EmbeddingsAnalyzer` instance with the embeddings being fit through the
estimator.

A notable feature implemented here is the analysis of the correlations between embedding
features and mathematical sequences, done in the `feature_to_sequence_analysis_df`
method. This is done by generating various mathematical sequences and then encoding them
using either:

- Direct encoding, for the ones that don't grow too much in value, like $\log_n$ or
  $n$. This simply means that the mathematical sequence vector the feature is tested
  against is produced by directly inserting the relative values, like $[log(0), log(1),
  log(2), \ldots]$
- One-hot encoding, for faster growing sequences. The indices that correspond to a value
  contained in sequence get the value 1, the others get the value 0.
- Gaussian-smoothed one-hot encoding, where the values are passed through a Gaussian
  filter to check for smoother feature detection.

As a result of the analysis, a dataframe is produced that provides data about features
and their respective correlations.

```python
def one_hot_encode(sequence, size):
    return np.isin(np.arange(size), sequence).astype(int)

def one_hot_gaussian_smooth(binary, sigma=2.0):
    return gaussian_filter1d(binary.astype(float), sigma=sigma)

def make_encoded_sequences(max_token: int, sigma: float = 2.0):
    encoded_sequences = {}

    direct_sequences = direct_encoded_base_sequences(max_token)
    for name, seq in direct_sequences.items():
        encoded_sequences[name, "direct"] = seq

    binary_sequences = binary_encoded_base_sequences(max_token)
    for name, seq in binary_sequences.items():
        one_hot = one_hot_encode(seq, max_token)
        encoded_sequences[name, "binary"] = one_hot
        encoded_sequences[name, "gauss"] = one_hot_gaussian_smooth(one_hot, sigma=sigma)

    return encoded_sequences
```

: Code for mathematical sequence encoding. {#lst:mathencoding}

## CLI

The end user can make use of the tools by loading a model through the CLI, which was
programmed using the `typer` library. It can be used to load model numerical and random
samples into the database through the command `embcli load <hf_model_id>`, which can
then be listed and consulted through the Marimo dashboard.



