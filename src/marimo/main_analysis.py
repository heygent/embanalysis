import marimo

__generated_with = "0.14.9"
app = marimo.App(
    app_title="Analyzing Numerical Embeddings",
    layout_file="layouts/main_analysis.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import duckdb
    from embanalysis.duckdb_loader import DuckDBLoader
    from embanalysis.constants import DB_PATH
    from embanalysis.analyzer import EmbeddingsAnalyzer

    from embanalysis.marimo_utils import component_plot_ui, plot_components_with_ui

    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from umap import UMAP
    return (
        DB_PATH,
        DuckDBLoader,
        EmbeddingsAnalyzer,
        PCA,
        TSNE,
        TruncatedSVD,
        UMAP,
        component_plot_ui,
        duckdb,
        mo,
        plot_components_with_ui,
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
    all_model_ids = models["model_id"].to_list()
    model_id_ui = mo.ui.dropdown(all_model_ids, searchable=True, value=all_model_ids[0])
    return (model_id_ui,)


@app.cell
def _(loader, model_id_ui):
    model_id = model_id_ui.value
    samples = loader.get_model_samples(model_id)
    model_id_ui
    return (samples,)


@app.cell
def _(EmbeddingsAnalyzer, samples):
    integers_analyzer = EmbeddingsAnalyzer.from_sample(samples["integers"])
    return (integers_analyzer,)


@app.cell
def _(mo):
    mo.md(r"""# Linear analysis - PCA and SVD""")
    return


@app.cell
def _(PCA, TruncatedSVD, integers_analyzer):
    svd_components = 100
    integers_svd = integers_analyzer.run_estimator(TruncatedSVD(svd_components))

    pca_components = 1000
    integers_pca = integers_analyzer.run_estimator(PCA(pca_components))

    return integers_pca, integers_svd, pca_components, svd_components


@app.cell
def _(mo):
    mo.md(r"""## Component correlations""")
    return


@app.cell
def _(integers_svd, mo):
    mo.ui.altair_chart(integers_svd.plot.correlation_heatmap())
    return


@app.cell
def _(integers_svd, mo):
    corr_table = mo.ui.table(integers_svd.top_correlations_df(), selection="single")
    corr_table
    return (corr_table,)


@app.cell
def _(component_plot_ui, corr_table, pca_components, svd_components):
    if len(corr_table.value) == 1:
        x = corr_table.value["Component1"].item()
        y = corr_table.value["Component2"].item()
    else:
        x = 0
        y = 1

    svd_ui = component_plot_ui(svd_components, x, y)
    pca_ui = component_plot_ui(pca_components, x, y)
    return pca_ui, svd_ui


@app.cell
def _(mo):
    mo.md(r"""## Component Plots""")
    return


@app.cell
def _(integers_pca, pca_ui, plot_components_with_ui):
    plot_components_with_ui(integers_pca, pca_ui)
    return


@app.cell
def _(integers_svd, plot_components_with_ui, svd_ui):
    plot_components_with_ui(integers_svd, svd_ui)
    return


@app.cell
def _(integers_pca, mo):
    mo.ui.plotly(integers_pca.plot.components_3d())
    return


@app.cell
def _(mo):
    mo.md(r"""## Variance""")
    return


@app.cell
def _(integers_svd):
    integers_svd.variance_df
    return


@app.cell
def _(integers_pca, mo):
    mo.vstack(
        [
            mo.ui.altair_chart(integers_pca.plot.explained_variance()),
            mo.ui.altair_chart(integers_pca.plot.cumulative_variance()),
        ],
    )
    return


@app.cell
def _(mo):
    mo.md("""# Non-linear analysis""")
    return


@app.cell
def _(TSNE, component_plot_ui, integers_analyzer):
    tsne_kwargs = dict(
        n_components=3,
        perplexity=75,
        learning_rate=50,
        early_exaggeration=20,
        random_state=42,
    )
    tsne = TSNE(**tsne_kwargs)

    integers_tsne = integers_analyzer.run_estimator(tsne)
    tsne_ui = component_plot_ui(tsne_kwargs["n_components"] - 1)
    return integers_tsne, tsne_kwargs, tsne_ui


@app.cell
def _(integers_tsne, plot_components_with_ui, tsne_ui):
    plot_components_with_ui(integers_tsne, tsne_ui)
    return


@app.cell
def _(integers_tsne):
    integers_tsne.plot.components_3d()
    return


@app.cell
def _(UMAP, component_plot_ui, integers_analyzer, tsne_kwargs):
    umap_kwargs = dict(
        n_components=3,
        # Increase from default 15 to preserve more global structure
        n_neighbors=50,
        # Decrease from default 0.1 for tighter local clusters
        min_dist=0.05,
        metric="cosine",
        # Increase from default 1.0 to spread out the visualization
        spread=1.5,
        # Increase to enhance local structure preservation
        local_connectivity=2,
        random_state=42,
    )
    umap = UMAP(**umap_kwargs)

    integers_umap = integers_analyzer.run_estimator(umap)
    umap_ui = component_plot_ui(tsne_kwargs["n_components"] - 1)
    return integers_umap, umap_ui


@app.cell
def _(integers_umap, plot_components_with_ui, umap_ui):
    plot_components_with_ui(integers_umap, umap_ui)
    return


@app.cell
def _(UMAP, component_plot_ui, integers_analyzer):
    umap_euc_kwargs = dict(
        n_components=2,
        # Increase from default 15 to preserve more global structure
        n_neighbors=50,
        # Decrease from default 0.1 for tighter local clusters
        min_dist=0.05,
        metric="euclidean",
        # Increase from default 1.0 to spread out the visualization
        spread=1.5,
        # Increase to enhance local structure preservation
        local_connectivity=2,
        random_state=42,
    )

    umap_euc = UMAP(**umap_euc_kwargs)

    integers_umap_euc = integers_analyzer.run_estimator(umap_euc)
    umap_euc_ui = component_plot_ui(umap_euc_kwargs["n_components"] - 1)
    return integers_umap_euc, umap_euc_ui


@app.cell
def _(integers_umap_euc, plot_components_with_ui, umap_euc_ui):
    plot_components_with_ui(integers_umap_euc, umap_euc_ui)
    return


@app.cell
def _(mo):
    mo.md(r"""# Component patterns""")
    return


@app.cell
def _(mo):
    current_analyzer_ui = mo.ui.dropdown()
    return


@app.cell
def _(integers_pca, mo):
    dim_corr_df = integers_pca.feature_to_sequence_analysis_df()
    dim_corr_table = mo.ui.table(dim_corr_df.reset_index(drop=True))
    dim_corr_table
    return


@app.cell
def _(mo, pca_components):
    pca_fourier_magnitude_ui = mo.ui.number(start=0, stop=pca_components, label="Component:")
    pca_fourier_magnitude_ui
    return (pca_fourier_magnitude_ui,)


@app.cell
def _(integers_pca, mo, pca_fourier_magnitude_ui):
    mo.vstack([
        mo.ui.altair_chart(
            integers_pca.plot.fourier_magnitude(pca_fourier_magnitude_ui.value)
        ),
        pca_fourier_magnitude_ui
    ], align='stretch')
    return


@app.cell
def _(integers_analyzer):
    integers_analyzer.analyze_correlations_with_properties()
    return


@app.cell
def _(integers_analyzer):
    integers_analyzer.plot.dimension_property_correlations(80)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
