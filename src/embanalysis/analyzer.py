from dataclasses import dataclass
from functools import cached_property
from typing import Literal
import altair as alt
import numpy as np
import pandas as pd
import re

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector

from embanalysis.sample_data import EmbeddingsSample, EmbeddingsSampleMeta, ReducedSampleMeta

default_props = {}


def wide_embeddings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert embeddings_df to wide format with one column per component."""

    expanded = pd.DataFrame(df['embeddings'].tolist())
    expanded.columns = [f"embeddings_{i}" for i in expanded.columns]
    return pd.concat([df.drop("embeddings", axis=1), expanded], axis=1)

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

    def run_estimator(self, estimator: BaseEstimator):
        """Run a scikit-learn estimator only on the embeddings columns of embeddings_df."""

        def verbose_feature_names_out(s1, s2):
            if s1 == "dim_reduction":
                return re.sub(r"\D+", "embeddings_", s2)
            return s2

        transformer = ColumnTransformer(
            [
                ("index", "passthrough", make_column_selector(dtype_include=["int", "object"])),
                (
                    "dim_reduction",
                    estimator,
                    make_column_selector(pattern="^embeddings_"),
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=verbose_feature_names_out
        ).set_output(transform="pandas")

        transformed_df = transformer.fit_transform(self.embeddings_df)

        return EmbeddingsAnalyzer(
            embeddings_df=transformed_df,
            meta=ReducedSampleMeta(
                original=self.meta,
                estimator=transformer.named_transformers_["dim_reduction"],
            ),
        )
    
    @cached_property
    def variance_df(self) -> pd.DataFrame:
        estimator = self.meta.estimator
        return pd.DataFrame(
            {
                "Component": np.arange(1, len(estimator.explained_variance_) + 1),
                "ExplainedVariance": estimator.explained_variance_,
                "ExplainedVarianceRatio": estimator.explained_variance_ratio_,
            }
        )

    @cached_property
    def embeddings_df_justdata(self) -> pd.DataFrame:
        return self.embeddings_df.drop(['token_id', 'token'], axis=1)


    def alt_title(self, **kwargs) -> alt.TitleParams:
        """Generate title parameters for altair charts."""
        return alt.TitleParams(
            text=self.meta.model_id,
            subtitle=self.meta.label(),
            fontSize=24,
            subtitleFontSize=16,
            anchor="middle",
            **kwargs,
        )
    
    def _base_component_chart(self, x_component: int, y_component: int) -> alt.Chart:
        chart = (
            alt.Chart(self.embeddings_df)
            .mark_circle(size=60, opacity=0.7)
            .properties(title=self.alt_title(), **default_props)
            .encode(
                x=alt.X(
                    f"embeddings_{x_component}",
                    title=f"Component {x_component + 1}",
                ),
                y=alt.Y(
                    f"embeddings_{y_component}",
                    title=f"Component {y_component + 1}",
                ),
            ).interactive()
        )

        legend = alt.Legend(
            labelLimit=150,
            symbolLimit=30,
            titleLimit=100,
            columns=1,
            padding=10
        )

        tooltip = [
            alt.Tooltip('token', title='Token'),
            alt.Tooltip(f'embeddings_{x_component}', title=f"Component {x_component + 1}", format=".5f"),
            alt.Tooltip(f'embeddings_{y_component}', title=f"Component {y_component + 1}", format=".5f"),
        ]

        return chart, tooltip, legend
    
    def plot_gradient(self, x_component: int = 0, y_component: int = 1) -> alt.Chart:
        chart, tooltip, legend = self._base_component_chart(x_component, y_component)

        return (
            chart.encode(
                color=alt.Color(
                    "token:Q",
                    scale=alt.Scale(scheme="viridis"),  # type: ignore
                    title="Token",
                    legend=legend,
                ),
                tooltip=tooltip,
            )
        )
    
    def plot_digit(
            self,
            x_component: int = 0,
            y_component: int = 1,
            digit_position: Literal[0, 1, 2] = 2,
        ) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components colored by digit position."""
        chart, tooltip, legend = self._base_component_chart(x_component, y_component)

        if digit_position not in [0, 1, 2]:
            raise ValueError("digit_position must be 0 (ones), 1 (tens), or 2 (hundreds)")

        position_label = ["Ones", "Tens", "Hundreds"][digit_position]

        return (
            chart.encode(
                color=alt.Color("digit:N", title=f"{position_label} Digit", legend=legend),
                tooltip=[*tooltip, alt.Tooltip("digit:N", title=f"{position_label} Digit")],
            ).transform_calculate(
                digit=f"floor(datum.token / pow(10, {digit_position})) % 10",
            )
        )
    
    def plot_digit_length(
            self,
            x_component: int = 0,
            y_component: int = 1,
        ) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components colored by digit length."""
        chart, tooltip, legend = self._base_component_chart(x_component, y_component)
        return (
            chart.encode(
                color=alt.Color("token_length:N", title="Token Length", legend=legend),
                tooltip=[*tooltip, alt.Tooltip("token_length:N", title="Token Length/Digits")],
                size=alt.Size(
                    "token_length:N", scale=alt.Scale(range=[200, 100, 30]), legend=None
                ),
            ).transform_calculate(
                token_length="length(toString(datum.token))",
            )
        )
    
    def plot_components(
            self,
            x_component: int = 0,
            y_component: int = 1,
            type: Literal['gradient', 'digit', 'digit_length'] = 'gradient',
            digit_position: int = 2,
        ) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components."""

        match type:
            case "gradient":
                return self.plot_gradient(x_component, y_component)
            case "digit":
                return self.plot_digit(x_component, y_component, digit_position)
            case "digit_length":
                return self.plot_digit_length(x_component, y_component)

    def plot_consecutive_distances(self) -> alt.Chart:
        """Create a chart showing distances between consecutive number embeddings."""
        # Get embedding columns
        embedding_cols = [col for col in self.embeddings_df.columns if col.startswith('embeddings_')]
        embedding_data = self.embeddings_df[embedding_cols].values
        
        consecutive_distances = np.linalg.norm(
            embedding_data[1:] - embedding_data[:-1], axis=1
        )
        distances_df = pd.DataFrame(
            {
                "Number": np.arange(1, len(consecutive_distances) + 1),
                "Distance": consecutive_distances,
            }
        )

        return (
            alt.Chart(distances_df)
            .mark_line()
            .encode(
                x=alt.X("Number:Q", title="Number n (distance is between n-1 and n)"),
                y=alt.Y("Distance:Q", title="Euclidean Distance"),
                tooltip=["Number", "Distance"],
            )
            .properties(
                title="Distances Between Consecutive Numbers",
                **default_props,
            )
            .interactive()
        )

    def plot_component_patterns(
        self, n_components: int = 5, n_values: int = 100, facet: bool = True
    ) -> alt.Chart:
        """Create a chart showing patterns in the top components."""
        # This function assumes there's a pca attribute - might need adjustment based on actual usage
        component_dfs = []
        for i in range(min(n_components, len(self.pca.components_))):
            df = pd.DataFrame(
                {
                    "Index": np.arange(min(n_values, len(self.pca.components_[i]))),
                    "Value": self.pca.components_[
                        i, : min(n_values, len(self.pca.components_[i]))
                    ],
                    "Component": f"Component {i + 1}",
                }
            )
            component_dfs.append(df)

        component_df = pd.concat(component_dfs)
        base_chart = (
            alt.Chart(component_df)
            .mark_line()
            .encode(
                x=alt.X("Index:Q", title="Index"),
                color=alt.Color("Component:N", title="Component"),
                tooltip=["Component", "Index", "Value"],
            )
            .properties(title="Component Patterns", **default_props)
            .interactive()
        )

        return (
            (
                base_chart.encode(y=alt.Y("Value:Q", title="Value"))
                .facet(row="Component:N")
                .resolve_scale(y="independent")
            )
            if facet
            else base_chart.encode(y=alt.Y("Value:Q", title="Value"))
        )
    
    def top_correlations_df(
        self, n_vectors: int = 20, min_correlation=0.01
    ) -> pd.DataFrame:
        df = self.embeddings_df_justdata
        n_vectors = min(n_vectors, df.shape[1])
        correlations = np.corrcoef(df.iloc[:, :n_vectors].T)

        # Get upper triangle indices (excluding diagonal)
        idx_i, idx_j = np.triu_indices(n_vectors, k=1)
        data = {
            "Component1": idx_i,
            "Component2": idx_j,
            "Correlation": correlations[idx_i, idx_j],
        }
        corr_df = pd.DataFrame(data)

        # Sort by absolute correlation descending
        corr_df = corr_df.reindex(
            corr_df["Correlation"].abs().sort_values(ascending=False).index
        )
        corr_df = corr_df[["Component1", "Component2", "Correlation"]]
        return corr_df[corr_df["Correlation"].abs() >= min_correlation]
    
    def top_correlated_components(self, n_vectors: int = 20, min_correlation=0.01) -> list[tuple[int, int]]:
        top_correlations_df = self.top_correlations_df(n_vectors, min_correlation=min_correlation)[['Component1', 'Component2']]
        return list(top_correlations_df.itertuples(index=False, name=None))

    def plot_top_correlated_components(self, n_vectors: int = 10) -> alt.Chart:
        """
        Plot the plot_by_digit_length chart for the top correlated component pairs.
        Arranges the plots in a grid with two plots per row.
        """
        corr_df = self.top_correlations_df(n_vectors)
        charts = []
        for _, row in corr_df.iterrows():
            i, j = int(row["Component1"]), int(row["Component2"])
            chart = self.plot_number_gradient(x_component=i, y_component=j).properties(
                title=f"Components {i + 1} vs {j + 1} (corr={row['Correlation']:.2f})"
            )
            charts.append(chart)
            chart = self.plot_by_digit_length(x_component=i, y_component=j).properties(
                title=f"Components {i + 1} vs {j + 1} (corr={row['Correlation']:.2f}) by digit length"
            )
            charts.append(chart)
        # Arrange charts in a grid with two plots per row
        rows = []
        for k in range(0, len(charts), 2):
            row = alt.hconcat(*charts[k : k + 2])
            rows.append(row)
        return alt.vconcat(*rows).properties(
            title=self.alt_title(),
        )

    def plot_correlation_heatmap(self, n_vectors: int = 20) -> alt.Chart:
        """Create a heatmap showing correlations between the top components, with value labels."""
        # Get embedding columns
        embedding_cols = [col for col in self.embeddings_df.columns if col.startswith('embeddings_')]
        n_vectors = min(n_vectors, len(embedding_cols))
        embedding_data = self.embeddings_df[embedding_cols[:n_vectors]].values
        correlations = np.corrcoef(embedding_data.T)

        corr_data = [
            {
                "Component1": i + 1,
                "Component2": j + 1,
                "Correlation": correlations[i, j],
            }
            for i in range(n_vectors)
            for j in range(n_vectors)
        ]
        df = pd.DataFrame(corr_data)

        heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("Component1:O", title="Component"),
                y=alt.Y("Component2:O", title="Component"),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                ),
                tooltip=["Component1", "Component2", "Correlation"],
            )
        )

        text = (
            alt.Chart(df)
            .mark_text(baseline="middle", fontSize=12)
            .encode(
                x=alt.X("Component1:O"),
                y=alt.Y("Component2:O"),
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "abs(datum.Correlation) > 0.5",
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        return (heatmap + text).properties(
            title=self.alt_title(),
            width=600,
            height=600,
        )

    def plot_explained_variance(self) -> alt.Chart:
        """Create a chart showing the explained variance per component."""
        return (
            alt.Chart(self.variance_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Component:Q", title="Component Index"),
                y=alt.Y("ExplainedVariance:Q", title="Explained Variance"),
                tooltip=["Component", "ExplainedVariance"],
            )
            .properties(
                title="Explained Variance Distribution",
                **default_props,
            )
            .interactive()
        )

    def plot_cumulative_variance(self, threshold: float = 0.9) -> alt.Chart:
        """Create a chart showing the cumulative explained variance."""
        # This assumes there's a variance_df attribute - might need adjustment
        chart = (
            alt.Chart(self.variance_df)
            .transform_window(
                CumulativeVariance="sum(ExplainedVarianceRatio)",
                Components="count()",
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("Components:Q", title="Number of Components"),
                y=alt.Y(
                    "CumulativeVariance:Q",
                    title="Cumulative Explained Variance",
                )
                .scale(domain=[0, 1])
                .axis(format=".0%"),
                tooltip=["Components:Q", "CumulativeVariance:Q"],
            )
            .properties(
                title="Cumulative Explained Variance",
                **default_props,
            )
            .interactive(bind_y=False)
        )

        threshold_rule = (
            alt.Chart().mark_rule(color="red", strokeDash=[4, 4]).encode(y=alt.Y(datum=threshold, title="Threshold"))
        )

        return chart + threshold_rule

    def plot_variance_overview(self) -> alt.Chart:
        """Create an overview plot combining explained and cumulative variance."""
        return alt.hconcat(
            self.plot_explained_variance(), self.plot_cumulative_variance()
        ).properties(title=self.alt_title())


