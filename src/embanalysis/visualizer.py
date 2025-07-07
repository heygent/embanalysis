from dataclasses import dataclass
from typing import Literal, Tuple
import altair as alt
import numpy as np
import pandas as pd

from embanalysis.analyzer import EmbeddingsAnalyzer
from embanalysis.feature_analysis import make_encoded_sequences, make_sequences

default_props = {
    # "width": 600,
}

@dataclass
class EmbeddingsVisualizer:
    """
    A class for creating visualizations of embeddings data.
    Handles all visualization functionality for EmbeddingsAnalyzer.
    """
    analyzer: EmbeddingsAnalyzer

    def __init__(self, analyzer: EmbeddingsAnalyzer, add_default_title: bool = True):
        self.analyzer = analyzer
        self.add_default_title = add_default_title

    def alt_title(self, **kwargs) -> alt.TitleParams:
        """Generate title parameters for altair charts."""
        if self.add_default_title:
            return alt.TitleParams(
                text=self.analyzer.meta.model_id,
                subtitle=self.analyzer.meta.label(),
                fontSize=24,
                subtitleFontSize=16,
                anchor="middle",
                **kwargs,
            )
        return None

    def _base_component_chart(self, x_component: int, y_component: int) -> Tuple[alt.Chart, list, alt.Legend]:
        """Create a base chart for component visualizations."""
        chart = (
            alt.Chart(self.analyzer.embeddings_df)
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
            )
            .interactive()
        )

        legend = alt.Legend(
            labelLimit=300, symbolLimit=100, titleLimit=500, columns=1, padding=10
        )

        tooltip = [
            alt.Tooltip("token", title="Token"),
            alt.Tooltip(
                f"embeddings_{x_component}",
                title=f"Component {x_component + 1}",
                format=".5f",
            ),
            alt.Tooltip(
                f"embeddings_{y_component}",
                title=f"Component {y_component + 1}",
                format=".5f",
            ),
        ]

        return chart, tooltip, legend

    def gradient(self, x_component: int = 0, y_component: int = 1) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components with a gradient color scheme."""
        chart, tooltip, legend = self._base_component_chart(x_component, y_component)

        return chart.encode(
            color=alt.Color(
                "token:Q",
                scale=alt.Scale(scheme="viridis"),  # type: ignore
                title="Token",
                legend=legend,
            ),
            tooltip=tooltip,
        )

    def strong_property_correlation_bar_chart(self) -> alt.Chart:
        """Create a bar chart showing properties with strong correlations to dimensions."""
        dim_corr_df = self.analyzer.feature_to_sequence_analysis_df()

        strong_corrs_df = dim_corr_df[
            (dim_corr_df["Correlation"].abs() > 0.20) & (dim_corr_df["P_Value"] < 0.05)
        ].copy()

        counts_per_property = strong_corrs_df["Property"].value_counts().reset_index()
        counts_per_property.columns = ["Property", "Count"]

        bar = (
            alt.Chart(counts_per_property)
            .mark_bar()
            .encode(
                x=alt.X("Property:N", title="Property", sort="-y"),
                y=alt.Y("Count:Q", title="Number of Strongly Correlated Dimensions"),
                color=alt.Color("Property:N", legend=None),
                tooltip=["Property:N", "Count:Q"],
            )
        )

        text = (
            alt.Chart(counts_per_property)
            .mark_text(
                align="center", baseline="bottom", dy=-2, fontSize=14, fontWeight="bold"
            )
            .encode(
                x=alt.X("Property:N", sort="-y"),
                y=alt.Y("Count:Q"),
                text=alt.Text("Count:Q"),
            )
        )

        return (bar + text).properties(
            title="Number of Strongly Correlated Dimensions (>0.20, p<0.05) per Property",
            width=400,
        )

    def digit(
        self,
        x_component: int = 0,
        y_component: int = 1,
        digit_position: Literal[0, 1, 2] = 2,
    ) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components colored by digit position."""
        chart, tooltip, legend = self._base_component_chart(x_component, y_component)

        if digit_position not in [0, 1, 2]:
            raise ValueError(
                "digit_position must be 0 (ones), 1 (tens), or 2 (hundreds)"
            )

        position_label = ["Ones", "Tens", "Hundreds"][digit_position]

        return chart.encode(
            color=alt.Color("digit:N", title=f"{position_label} Digit", legend=legend),
            tooltip=[*tooltip, alt.Tooltip("digit:N", title=f"{position_label} Digit")],
        ).transform_calculate(
            digit=f"floor(datum.token / pow(10, {digit_position})) % 10",
        )

    def digit_length(
        self,
        x_component: int = 0,
        y_component: int = 1,
    ) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components colored by digit length."""
        chart, tooltip, legend = self._base_component_chart(x_component, y_component)

        if self.analyzer.meta.original.tag == "random":
            color_type = "quantitative"
        else:
            color_type = "nominal"

        return chart.encode(
            color=alt.Color(
                "token_length", title="Token Length", legend=legend, type=color_type
            ),
            tooltip=[
                *tooltip,
                alt.Tooltip(
                    "token_length", title="Token Length/Digits", type=color_type
                ),
            ],
        ).transform_calculate(
            token_length="length(toString(datum.token))",
        )

    def components(
        self,
        x_component: int = 0,
        y_component: int = 1,
        plot_type: Literal["gradient", "digit", "digit_length"] = "gradient",
        digit_position: int = 2,
    ) -> alt.Chart:
        """Create a 2D scatter plot of two embeddings components."""

        if self.analyzer.meta.original.tag == "random":
            return self.digit_length(x_component, y_component)

        match plot_type:
            case "gradient":
                return self.gradient(x_component, y_component)
            case "digit":
                return self.digit(x_component, y_component, digit_position)
            case "digit_length":
                return self.digit_length(x_component, y_component)
    
    def components_3d(self, x_component: int = 0, y_component: int = 1, z_component: int = 2) -> alt.Chart:
        pass
    
    def feature(self, component: int = 0) -> alt.Chart:
        """Create a chart showing the values of a single component."""
        embedding_col = f"embeddings_{component}"
        return (
            alt.Chart(self.analyzer.embeddings_df)
            .mark_line()
            .encode(
                x=alt.X("token:Q", title="Token ID"),
                y=alt.Y(embedding_col, title=f"Component {component} Value"),
                tooltip=["token", embedding_col],
            )
            .properties(
                title=f"Component {component} Values",
                **default_props,
            )
            .interactive(bind_y=False)
        )
    
    def feature_with_discrete_sequences(self, feature: int, sequence_names: list[str]) -> alt.Chart:
        """Create a chart showing the values of a specific sequence."""
        all_sequences_df = pd.DataFrame(make_encoded_sequences(len(self.analyzer.embeddings_df)))
        all_sequences_df.columns = all_sequences_df.columns.map(lambda x: "/".join(x))
        all_sequences_df.reset_index(inplace=True)
        
        chart = self.feature(feature)
        for sequence in sequence_names:
            chart += alt.Chart(all_sequences_df).mark_rule(strokeDash=[5, 5]).encode(x='index', y=f'{sequence}/binary:Q')
        
        return chart
    
    def fourier_magnitude(self, component: int = 0, max_y: float = 20.0) -> alt.Chart:
        """Create a chart showing the Fourier magnitude of a specific component."""

        fourier_df = self.analyzer.fourier_dimension_df(component).reset_index()

        return (
            alt.Chart(fourier_df)
            .mark_line()
            .encode(
                x=alt.X('index'),
                y=alt.Y(f"magnitude:Q", title=f"Magnitude of Component {component}", scale=alt.Scale(domain=[0, max_y])),
                tooltip=["frequency:Q", f"magnitude:Q"],
            )
            .properties(
                title=f"Fourier Magnitude of Component {component}",
                **default_props,
            )
            .interactive(bind_x=False)
        )
        

    def consecutive_distances(self) -> alt.Chart:
        """Create a chart showing distances between consecutive number embeddings."""
        # Get embedding columns
        embedding_cols = [
            col for col in self.analyzer.embeddings_df.columns if col.startswith("embeddings_")
        ]
        embedding_data = self.analyzer.embeddings_df[embedding_cols].values

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

    def component_patterns(
        self, n_components: int = 5, n_values: int = 100, facet: bool = True
    ) -> alt.Chart:
        """Create a chart showing patterns in the top components."""
        # This assumes the analyzer has a pca attribute - might need adjustment
        if not hasattr(self.analyzer, 'pca'):
            raise AttributeError("Analyzer does not have a 'pca' attribute")
            
        component_dfs = []
        for i in range(min(n_components, len(self.analyzer.pca.components_))):
            df = pd.DataFrame(
                {
                    "Index": np.arange(min(n_values, len(self.analyzer.pca.components_[i]))),
                    "Value": self.analyzer.pca.components_[
                        i, : min(n_values, len(self.analyzer.pca.components_[i]))
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

    def correlation_heatmap(self, n_vectors: int = 20) -> alt.Chart:
        """Create a heatmap showing correlations between the top components, with value labels."""
        # Get embedding columns
        embedding_cols = [
            col for col in self.analyzer.embeddings_df.columns if col.startswith("embeddings_")
        ]
        n_vectors = min(n_vectors, len(embedding_cols))
        embedding_data = self.analyzer.embeddings_df[embedding_cols[:n_vectors]].values
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

    def dimension_property_correlations(
        self, top_n: int = 20, min_abs_correlation: float = 0.1
    ) -> alt.Chart:
        """
        Create a heatmap showing correlations between embedding dimensions and numerical properties.

        Args:
            top_n: Number of top correlations to show
            min_abs_correlation: Minimum absolute correlation to display

        Returns:
            Altair chart showing the correlation heatmap
        """
        corr_df = self.analyzer.feature_to_sequence_analysis_df()

        # Filter by minimum correlation and take top N
        filtered_df = corr_df[corr_df["Correlation"].abs() >= min_abs_correlation].head(
            top_n
        )

        # Create heatmap
        heatmap = (
            alt.Chart(filtered_df)
            .mark_rect()
            .encode(
                x=alt.X("Property:N", title="Property", sort="-y"),
                y=alt.Y("Dimension:O", title="Embedding Dimension", sort="-x"),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                    title="Correlation",
                ),
                tooltip=[
                    "Dimension:O",
                    "Property:N",
                    alt.Tooltip("Correlation:Q", format=".3f"),
                    alt.Tooltip("P_Value:Q", format=".2e"),
                ],
            )
        )

        # Add text labels for correlations
        text = (
            alt.Chart(filtered_df)
            .mark_text(baseline="middle", fontSize=10, fontWeight="bold")
            .encode(
                x=alt.X("Property:N"),
                y=alt.Y("Dimension:O"),
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "abs(datum.Correlation) > 0.5",
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        return (heatmap + text).properties(
            title=self.alt_title(),  # text="Dimension-Property Correlations"),
            width=400,
            height=max(300, len(filtered_df["Dimension"].unique()) * 20),
        )

    def explained_variance(self) -> alt.Chart:
        """Create a chart showing the explained variance per component."""
        return (
            alt.Chart(self.analyzer.variance_df)
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

    def cumulative_variance(self, threshold: float = 0.9) -> alt.Chart:
        """Create a chart showing the cumulative explained variance."""
        chart = (
            alt.Chart(self.analyzer.variance_df)
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
            alt.Chart()
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y=alt.Y(datum=threshold, title="Threshold"))
        )

        return chart + threshold_rule

    def variance_overview(self) -> alt.Chart:
        """Create an overview plot combining explained and cumulative variance."""
        return alt.hconcat(
            self.explained_variance(), self.cumulative_variance()
        ).properties(title=self.alt_title())

    def component_loadings(self, n_components: int = 5) -> alt.Chart:
        """Plot the loadings (eigenvectors) showing feature contributions to each component."""
        estimator = self.analyzer.meta.estimator

        # Create DataFrame with loadings
        loading_dfs = []
        for i in range(min(n_components, estimator.components_.shape[0])):
            df = pd.DataFrame(
                {
                    "Feature_Index": np.arange(len(estimator.components_[i])),
                    "Loading": estimator.components_[i],
                    "Component": f"Component {i + 1}",
                    "Magnitude": np.abs(estimator.components_[i]),
                }
            )
            loading_dfs.append(df)

        loadings_df = pd.concat(loading_dfs)

        return (
            alt.Chart(loadings_df)
            .mark_line()
            .encode(
                x=alt.X("Feature_Index:Q", title="Original Feature Index"),
                y=alt.Y("Loading:Q", title="Loading Value"),
                color=alt.Color("Component:N"),
                tooltip=["Component", "Feature_Index", "Loading:Q"],
            )
            .facet(row="Component:N")
            .resolve_scale(y="independent")
            .properties(
                title="Principal Component Loadings (Eigenvectors)",
                width=400,
                height=100,
            )
        )

    def top_correlated_components(
        self, n_vectors: int = 10, corr_df=None
    ) -> alt.Chart:
        """
        Plot the plot_by_digit_length chart for the top correlated component pairs.
        Arranges the plots in a grid with two plots per row.
        """
        if corr_df is None:
            corr_df = self.analyzer.top_correlations_df(n_vectors)
        charts = []
        for _, row in corr_df.iterrows():
            i, j = int(row["Component1"]), int(row["Component2"])
            chart = self.components(
                plot_type="gradient", x_component=i, y_component=j
            ).properties(
                title=f"Components {i + 1} vs {j + 1} (corr={row['Correlation']:.2f})"
            )
            charts.append(chart)
            chart = self.components(
                plot_type="digit_length", x_component=i, y_component=j
            ).properties(
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
