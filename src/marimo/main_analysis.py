

import marimo

__generated_with = "0.13.0"
app = marimo.App(app_title="Analyzing Numerical Embeddings")


@app.cell
def _():
    import marimo as mo
    import duckdb
    from embanalysis.duckdb_loader import DuckDBLoader
    from embanalysis.constants import DB_PATH
    from embanalysis.analyzer import EmbeddingsAnalyzer

    from sklearn.decomposition import PCA, TruncatedSVD
    return (
        DB_PATH,
        DuckDBLoader,
        EmbeddingsAnalyzer,
        PCA,
        TruncatedSVD,
        duckdb,
        mo,
    )


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
    model_id_ui = mo.ui.dropdown(all_model_ids, searchable=True, value=all_model_ids[0])
    return (model_id_ui,)


@app.cell
def _(loader, model_id_ui):
    model_id = model_id_ui.value
    samples = loader.get_model_samples(model_id)
    return (samples,)


@app.cell
def _(EmbeddingsAnalyzer, samples):
    integers_analyzer = EmbeddingsAnalyzer.from_sample(samples['integers'])
    return (integers_analyzer,)


@app.cell
def _(PCA, integers_analyzer, mo):
    pca_components = 1000
    integers_pca = integers_analyzer.run_estimator(PCA(pca_components))
    pca_x_ui = mo.ui.number(start=0, stop=pca_components, value=0, label="X Component")
    pca_y_ui = mo.ui.number(start=0, stop=pca_components, value=1, label="Y Component")
    pca_plot_type = mo.ui.dropdown({
        "Token value gradient": ("gradient",),
        "Digit length": ("digit_length",),
        "Ones Digit": ("digit", 0),
        "Tens Digit": ("digit", 1),
        "Hundreds Digit": ("digit", 2),
    }, label="Coloring", value="Token value gradient")
    return integers_pca, pca_plot_type, pca_x_ui, pca_y_ui


@app.cell
def _():
    return


@app.cell
def _(integers_pca, mo, pca_plot_type, pca_x_ui, pca_y_ui):
    mo.vstack([
        mo.ui.altair_chart(integers_pca.plot_components(
            pca_x_ui.value,
            pca_y_ui.value,
            *pca_plot_type.value
        )),
        mo.hstack([
            mo.vstack([pca_x_ui, pca_y_ui], align="stretch"),
            pca_plot_type
        ])
    ], align="start")
    return


@app.cell
def _(TruncatedSVD, integers_analyzer, mo):
    svd_components = 100
    integers_svd = integers_analyzer.run_estimator(TruncatedSVD(svd_components))
    svd_x_ui = mo.ui.number(start=0, stop=svd_components, value=0, label="X Component")
    svd_y_ui = mo.ui.number(start=0, stop=svd_components, value=1, label="Y Component")
    return integers_svd, svd_x_ui, svd_y_ui


@app.cell
def _(integers_svd, mo, svd_x_ui, svd_y_ui):
    mo.vstack([
        mo.ui.altair_chart(integers_svd.plot(
            svd_x_ui.value,
            svd_y_ui.value
        )),
        mo.hstack([svd_x_ui, svd_y_ui], align="stretch")
    ])
    return


@app.cell
def _(integers_pca, integers_svd, mo, pca_x_ui, pca_y_ui):
    mo.vstack([
        mo.hstack([
            mo.ui.altair_chart(integers_pca.plot(
                pca_x_ui.value,
                pca_y_ui.value
            )),
            mo.ui.altair_chart(integers_svd.plot(
                pca_x_ui.value,
                pca_y_ui.value
            )),
        ], widths="equal"),
        mo.vstack([pca_x_ui, pca_y_ui], justify="start")
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
