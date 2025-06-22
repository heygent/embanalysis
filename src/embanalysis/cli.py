from pathlib import Path
import typer
from typing_extensions import Annotated

from embanalysis.constants import HF_MODEL_ALIASES, DB_PATH
from embanalysis.duckdb_loader import DuckDBLoader
from embanalysis.sampler import HFEmbeddingsSampler

app = typer.Typer(no_args_is_help=True)


@app.command()
def load(
    model_id: str,
    db_path: Annotated[Path, typer.Option("--db-path", "-d", help="Path to the DuckDB database file")] = DB_PATH
):
    """Load embeddings from a Hugging Face model into a DuckDB database."""

    if model_id in HF_MODEL_ALIASES:
        model_id = HF_MODEL_ALIASES[model_id]

    sampler = HFEmbeddingsSampler.from_model(model_id)
    loader = DuckDBLoader(db_path)

    for sample, meta in (sampler.single_token_integers(), sampler.random()):
        loader.store_sample(sample, meta)
    

def main():
    app()


if __name__ == "__main__":
    main()
