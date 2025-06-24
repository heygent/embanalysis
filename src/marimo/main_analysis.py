

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium", app_title="Analyzing Numerical Embeddings")


@app.cell
def _():
    import marimo as mo
    import duckdb
    from embanalysis.duckdb_loader import DuckDBLoader
    from embanalysis.constants import DB_PATH
    from embanalysis.plots import EmbeddingsAnalyzer

    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from umap import UMAP
    return DB_PATH, DuckDBLoader, EmbeddingsAnalyzer, PCA, duckdb, mo


@app.cell
def _(DB_PATH, DuckDBLoader, duckdb):
    conn = duckdb.connect(DB_PATH, read_only=True)
    loader = DuckDBLoader(conn)
    return conn, loader


@app.cell
def _(conn, embeddings, mo):
    models = mo.sql(
        f"""
        SELECT DISTINCT model_id FROM embeddings;
        """,
        output=False,
        engine=conn
    )
    return (models,)


@app.cell
def _(mo, models):
    all_model_ids = models['model_id'].to_list()
    model_id = mo.ui.dropdown(all_model_ids, searchable=True)
    model_id
    return (model_id,)


@app.cell
def _(loader, model_id):
    samples = loader.get_model_samples(model_id.value)
    samples
    return (samples,)


@app.cell
def _(EmbeddingsAnalyzer, samples):
    analyzer = EmbeddingsAnalyzer.from_sample(samples['integers'])
    return (analyzer,)


@app.cell
def _(PCA, analyzer):
    pca = analyzer.run_estimator(PCA(1000))
    return (pca,)


@app.cell
def _(pca):
    pca.embeddings_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
