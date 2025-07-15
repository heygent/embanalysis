import marimo

__generated_with = "0.14.9"
app = marimo.App(
    app_title="Concrete Numeric Representations in LLM Embeddings",
    layout_file="layouts/presentation2.slides.json",
)

with app.setup:
    import marimo as mo
    import duckdb
    from embanalysis.duckdb_loader import DuckDBLoader
    from embanalysis.constants import DB_PATH
    from embanalysis.analyzer import EmbeddingsAnalyzer

    from embanalysis.marimo_utils import (
        plot_components_with_type_ui,
        plot_3d_components_with_type_ui,
        plot_type_ui,
    )

    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from umap import UMAP


@app.cell
def _():
    conn = duckdb.connect(DB_PATH, read_only=True)
    loader = DuckDBLoader(conn)
    return conn, loader


@app.cell
def _():
    mo.md(
        r"""
    # Model Analysis

    We take two models in consideration:

    - **allenai/OLMo-2-1124-7B**  
      OLMo2 is an open-source model built with replicability and research in mind, well-documented and with checkpoints at various stages available  
      **Training data**: 5T tokens


    - **meta-llama/Llama-3.2-1B-Instruct**  
      One of the most popular open source models, trained on a lot more data  
      **Training data**: 15T tokens
    """
    )
    return


@app.cell
def _(conn, embeddings):
    all_model_ids = mo.sql(
        f"""
        SELECT DISTINCT model_id FROM embeddings;
        """,
        engine=conn,
    )["model_id"].to_list()
    return


@app.cell
def _(loader):
    model_id = "allenai/OLMo-2-1124-7B"
    samples = loader.get_model_samples(model_id)
    return (samples,)


@app.cell
def _(samples):
    integers_analyzer = EmbeddingsAnalyzer.from_sample(samples["integers"])
    return (integers_analyzer,)


@app.cell
def _(integers_analyzer):
    svd_components = 100
    integers_svd = integers_analyzer.run_estimator(TruncatedSVD(svd_components))

    pca_components = 1000
    integers_pca = integers_analyzer.run_estimator(PCA(pca_components))
    return integers_pca, integers_svd


@app.cell
def _():
    type_ui = plot_type_ui()
    return (type_ui,)


@app.cell
def _(integers_pca, type_ui):
    mo.hstack(
        [
            plot_components_with_type_ui(integers_pca, 0, 1, type_ui),
            mo.md("""
    ## OLMo - PCA

    - Shows a U-curve with a smooth gradient between the numbers
    - Top-right corner has all one and two-digit numbers
        - The gradient is smooth and the position corresponds with the lowest numbers
    - PCA destroys structure (see SVD next)

        """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(integers_svd, type_ui):
    mo.hstack(
        [
            plot_components_with_type_ui(integers_svd, 0, 1, type_ui),
            mo.md("""
    ## OLMo - SVD

    - Triangular structures are much clearer
    - Self-repeating pattern on the basis of the number of digits
        - Very clear clustering
    - Clustering is also visible (better as the significance of the digit grows) for digits in all positions
        """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(integers_svd, type_ui):
    mo.hstack(
        [
            plot_3d_components_with_type_ui(integers_svd, 0, 1, 2, type_ui),
            mo.md("""
    ## OLMo - SVD 3D

    - Hint of a self-looping structure
    - Zero and lower numbers gravitate towards the center
        """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(integers_pca):
    mo.hstack(
        [
            mo.ui.altair_chart(
                integers_pca.plot.explained_variance()
                & integers_pca.plot.cumulative_variance()
            ),
            mo.md("""
    ## OLMo - Cumulative and Explained Variance

    - Sharp drop in the explained variance by component after 30-50 components
    - 90% cumulative explained variance reached after 600 components
        - much lower than the 4096 embedding dimensions OLMo has
        - number representations lie on a low dimensional manifold in the embedding space
    """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(integers_analyzer):
    tsne_kwargs = dict(
        n_components=2,
        perplexity=75,
        learning_rate=50,
        early_exaggeration=20,
        random_state=42,
    )
    tsne = TSNE(**tsne_kwargs)

    integers_tsne = integers_analyzer.run_estimator(tsne)
    return integers_tsne, tsne_kwargs


@app.cell
def _(integers_tsne, type_ui):
    mo.hstack(
        [
            plot_components_with_type_ui(integers_tsne, 0, 1, type_ui),
            mo.md("""
    ## OLMo - t-SNE

    - t-SNE reveals circular/spiral patterns where numerical tokens form branches that gradually increase, turn around center, then return to start 
    - Numbers follow continuous paths that preserve numerical ordering relationships in the embedding space
    - When colored by hundreds digit, distinct branches/filaments cluster neatly by their hundreds group
        """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(integers_analyzer):
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
    return integers_umap, umap_kwargs


@app.cell
def _(integers_umap, type_ui):
    mo.hstack(
        [
            plot_3d_components_with_type_ui(integers_umap, 0, 1, 2, type_ui),
            mo.md("""
    ## OLMo - UMAP

    - Little segments go in upward progression, almost like a ladder
        - The two outliers are double-digit and 1XX
    - Mantains a theme where non-linear projections seem to "open up" or "unfold" structures
    - A lot of 00-ending numbers are clustered on a single point
        - Reminds of t-SNE radial structure
        """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(integers_umap_euc, plot_components_with_ui, umap_euc_ui):
    plot_components_with_ui(integers_umap_euc, umap_euc_ui)
    return


@app.cell
def _():
    mo.md(
        r"""
    # Number Semantics

    - We also have an opportunity to check semantics in a way that crosses the symbolic barrier.
    - We can relate the symbols themselves (0, 1, 2, ..., 999) to the **numeric components that constitute the embeddings**
    - We explore this idea by checking individual features for all the integer embeddings and  their correlation with important mathematical sequences:
        - $n_i = i$ (the numbers themselves)
        - $n_i = \log(i)$
        - Prime numbers
        - Fibonacci numbers
        - Triangular numbers
    """
    )
    return


@app.cell
def _(integers_analyzer):
    dim_corr_df = integers_analyzer.feature_to_sequence_analysis_df()
    dim_corr_table = mo.ui.table(dim_corr_df.reset_index(drop=True))

    mo.hstack(
        [
            mo.vstack([
                dim_corr_table,
                mo.ui.altair_chart(
                integers_analyzer.plot.strong_property_correlation_bar_chart()
                )
            ]),
            mo.md("""
    - We find very high correlation for a lot of components for measures that encode for magnitude, as it would be expected. 
    - We also find correlation for more interesting sequences, in particular Fibonacci numbers
        - Fibonacci correlation in particular can be a general hint towards helicoidal and self-similar structures
        """),
        ]
    )
    return


@app.cell
def _(loader):
    llama_id = "meta-llama/Llama-3.2-1B-Instruct"
    llama_samples = loader.get_model_samples(llama_id)
    llama_integers_analyzer = EmbeddingsAnalyzer.from_sample(
        llama_samples["integers"]
    )
    llama_svd_components = 100
    llama_integers_svd = llama_integers_analyzer.run_estimator(
        TruncatedSVD(llama_svd_components)
    )

    llama_pca_components = 1000
    llama_integers_pca = llama_integers_analyzer.run_estimator(
        PCA(llama_pca_components)
    )
    return llama_integers_analyzer, llama_integers_pca, llama_integers_svd


@app.cell
def _(llama_integers_pca, type_ui):
    mo.hstack(
        [
            plot_components_with_type_ui(llama_integers_pca, 0, 1, type_ui),
            mo.md("""
    ## Llama - PCA

    -  Much more pronounced division and separation between numbers of different digit lengths than in OLMo
    -  Recursive structure based on digit count is immediately visible - same patterns repeat for different digit sizes
        -  Remarkable how the same structural patterns for different digit counts persist across different model architectures
            """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(llama_integers_svd, type_ui):
    mo.hstack(
        [
            plot_3d_components_with_type_ui(llama_integers_svd, 0, 1, 2, type_ui),
            mo.md("""
    ## Llama - SVD

    - 3D projection shows how the same structure repeats at different distances and angles of rotation
    -  Token length coloring reveals distinct groupings maintaining the structural patterns across different digit counts
            """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(llama_integers_pca):
    mo.hstack(
        [
            mo.ui.altair_chart(
                llama_integers_pca.plot.explained_variance()
                & llama_integers_pca.plot.cumulative_variance()
            ),
            mo.md("""
    ## Llama - Cumulative and Explained Variance

    - 90% cumulative variance reached after ~500 components
    - Slightly more dimensions needed than OLMo to capture similar variance
    """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(llama_integers_analyzer, tsne_kwargs):
    llama_tsne = TSNE(**tsne_kwargs)
    llama_integers_tsne = llama_integers_analyzer.run_estimator(llama_tsne)
    return (llama_integers_tsne,)


@app.cell
def _(llama_integers_tsne, type_ui):
    mo.hstack(
        [
            plot_components_with_type_ui(llama_integers_tsne, 0, 1, type_ui),
            mo.md("""
    ## Llama - t-SNE

    - This is one of the best visualizations of possible helicoidal structures happening in models
    - Clustering by hundreds, in different filaments that seem uncoiled from a spiral
    - Multiples of 100 clustered in a single point, like in the OLMo UMAP visualization
            """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(llama_integers_analyzer, type_ui, umap_kwargs):
    llama_umap = UMAP(**umap_kwargs)
    llama_integers_umap = llama_integers_analyzer.run_estimator(llama_umap)
    plot_components_with_type_ui(llama_integers_umap, 0, 1, type_ui)
    return


@app.cell
def _(llama_integers_analyzer, type_ui, umap_euc_kwargs):
    llama_umap_euc = UMAP(**umap_euc_kwargs)
    llama_integers_umap_euc = llama_integers_analyzer.run_estimator(llama_umap_euc)
    plot_components_with_type_ui(llama_integers_umap_euc, type_ui)
    return


@app.cell
def _(llama_integers_analyzer):
    llama_dim_corr_df = llama_integers_analyzer.feature_to_sequence_analysis_df()
    llama_dim_corr_table = mo.ui.table(llama_dim_corr_df.reset_index(drop=True))
    return (llama_dim_corr_table,)


@app.cell
def _(llama_dim_corr_table, llama_integers_analyzer):
    mo.hstack(
        [
            mo.vstack([
                llama_dim_corr_table,
                mo.ui.altair_chart(
                llama_integers_analyzer.plot.strong_property_correlation_bar_chart()
                )
            ]),
            mo.md("""
    - Similar high correlation with magnitude-related sequences
    - Fewer significant correlations with special sequences compared to OLMo
    - Slightly lower correlation coefficients overall
    - Patterns suggest less explicit encoding of mathematical properties
        """),
        ]
    )
    return


@app.cell
def _(llama_integers_tsne, type_ui):
    mo.hstack(
        [
            plot_components_with_type_ui(llama_integers_tsne, 0, 1, type_ui),
            mo.md("""
    ## Llama - t-SNE

    - This is one of the best visualizations of possible helicoidal structures happening in models
    - Clustering by hundreds, in different filaments that seem uncoiled from a spiral
    - Multiples of 100 clustered in a single point, like in the OLMo UMAP visualization
            """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _(llama_integers_analyzer):
    llama_tsne_3d_kwargs = dict(
        n_components=3,
        perplexity=30,
        learning_rate='auto',
        max_iter=5000,
        min_grad_norm=1e-8,
        method='barnes_hut',
        angle=0.3,
        early_exaggeration=4,
        random_state=42,
    )

    llama_tsne_3d = TSNE(
        **llama_tsne_3d_kwargs
    
    )
    llama_integers_tsne_3d = llama_integers_analyzer.run_estimator(llama_tsne_3d)
    return (llama_integers_tsne_3d,)


@app.cell
def _(llama_integers_tsne_3d, type_ui):
    mo.hstack(
        [
            plot_3d_components_with_type_ui(llama_integers_tsne_3d, 0, 1, 2, type_ui),
            mo.md("""
    ## Llama - t-SNE

    - This is one of the best visualizations of possible helicoidal structures happening in models
    - Clustering by hundreds, in different filaments that seem uncoiled from a spiral
    - Multiples of 100 clustered in a single point, like in the OLMo UMAP visualization
            """),
        ],
        widths=[0.6, 0.4],
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    # Conclusions

    - We have compelling reasons to explore the geometric structure of numerical embeddings
    - Their shape could tell us a lot about the way LLMs organize information
        - If there are general information-theory principles behind those reasons, these might inform us on models beyond LLMs
        - As the properties we observe seem to scale with model size, it might be the case
        - More experiments on more models are needed to definitely establish causal connections
    - We find a lot of possible ways to move this forward:
          - Searching for more precise convergence patterns
          - Looking at more connections between sequences and features
          - Looking for functions that fit well the generation of number embeddings
  
      
      
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    # Thank you

    """
    )
    return


if __name__ == "__main__":
    app.run()
