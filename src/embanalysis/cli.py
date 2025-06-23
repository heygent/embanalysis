from pathlib import Path
import typer
import rich
from typing_extensions import Annotated
import contextlib

from embanalysis.constants import DEFAULT_SEED, HF_MODEL_ALIASES, DB_PATH
from embanalysis.duckdb_loader import DuckDBLoader
from embanalysis.sampler import HFEmbeddingsSampler


def df_to_table(df, title=None) -> rich.table.Table:
    # ensure dataframe contains only string values
    df = df.astype(str)

    table = rich.table.Table(title=title)
    for col in df.columns:
        table.add_column(col)
    for row in df.values:
        with contextlib.suppress(rich.errors.NotRenderableError):
            table.add_row(*row)
    return table


app = typer.Typer(no_args_is_help=True)


def alias_to_model_id(model_id: str) -> str:
    return HF_MODEL_ALIASES.get(model_id, model_id)


ModelIDArg = Annotated[
    str,
    typer.Argument(help="Hugging Face model ID or alias", callback=alias_to_model_id),
]
DBPathOption = Annotated[
    Path, typer.Option("--db-path", "-d", help="Path to the DuckDB database file")
]

SeedOption = Annotated[
    int, typer.Option("--seed", "-s", help="Random seed for sampling")
]


@app.command()
def load(
    model_id: ModelIDArg,
    db_path: DBPathOption = DB_PATH,
    seed: SeedOption = DEFAULT_SEED,
):
    """Load embeddings from a Hugging Face model into the DuckDB database."""

    sampler = HFEmbeddingsSampler.from_model(model_id)

    with DuckDBLoader(db_path) as loader:
        for df, meta in (
            sampler.single_token_integers(),
            sampler.random(seed=seed),
        ):
            loader.store_sample(model_id, df, meta)


@app.command()
def init(
    db_path: DBPathOption = DB_PATH,
):
    """Initializes the DuckDB database schema."""
    with DuckDBLoader(db_path) as loader:
        loader.init_db()


@app.command()
def reset(db_path: DBPathOption = DB_PATH):
    """Deletes and reinitializes the database."""
    with DuckDBLoader(db_path) as loader:
        loader.drop_db()
        loader.init_db()


@app.command()
def shell(
    db_path: DBPathOption = DB_PATH,
):
    """Starts an interactive DuckDB shell."""
    import os

    os.execvp("duckdb", ["duckdb", str(db_path)])


list = typer.Typer()


@list.command()
def samples(
    db_path: DBPathOption = DB_PATH,
):
    """Lists all samples in the DuckDB database."""
    with DuckDBLoader(db_path) as loader:
        samples = loader.list_samples()
        rich.print(df_to_table(samples))


@list.command()
def models(
    db_path: DBPathOption = DB_PATH,
):
    """Lists all models in the DuckDB database."""
    with DuckDBLoader(db_path) as loader:
        models = loader.list_models()
        for (model,) in models:
            print(model)


app.add_typer(list, name="list", help="Show available data in the database")


def main():
    app()


if __name__ == "__main__":
    main()
