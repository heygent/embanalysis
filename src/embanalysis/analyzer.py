from dataclasses import dataclass
from functools import cached_property
from typing import Self
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.base import BaseEstimator

from embanalysis.feature_analysis import make_sequences

from embanalysis.sample_data import (
    EmbeddingsSample,
    EmbeddingsSampleMeta,
    ReducedSampleMeta,
)


def wide_embeddings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert embeddings_df to wide format with one column per component."""

    expanded = pd.DataFrame(df["embeddings"].tolist())
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
    
    @cached_property
    def plot(self):
        # local import to avoid circular dependency
        from embanalysis.visualizer import EmbeddingsVisualizer
        return EmbeddingsVisualizer(self, add_default_title=True)
    
    def run_estimator(self, estimator: BaseEstimator) -> Self:
        """Run a scikit-learn estimator only on the embeddings columns of embeddings_df."""

        # Select embeddings columns
        embedding_cols = [
            col for col in self.embeddings_df.columns if col.startswith("embeddings_")
        ]
        other_cols = [
            col
            for col in self.embeddings_df.columns
            if not col.startswith("embeddings_")
        ]

        # Fit estimator on embeddings
        embeddings_data = self.embeddings_df[embedding_cols].values
        transformed = estimator.fit_transform(embeddings_data)

        # If transformed is 1D, make it 2D
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)

        # Find number of columns from transformed data
        n_components = transformed.shape[1]

        transformed_cols = [f"embeddings_{i}" for i in range(n_components)]
        transformed_df = pd.DataFrame(
            transformed, columns=transformed_cols, index=self.embeddings_df.index
        )

        # Concatenate with other columns
        result_df = pd.concat([self.embeddings_df[other_cols], transformed_df], axis=1)

        return EmbeddingsAnalyzer(
            embeddings_df=result_df,
            meta=ReducedSampleMeta(
                original=self.meta,
                estimator=estimator,
            ),
        )

    @cached_property
    def variance_df(self) -> pd.DataFrame:
        """Get variance data from the estimator as a DataFrame."""
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
        """Get just the data columns (exclude token_id and token)."""
        return self.embeddings_df.drop(["token_id", "token"], axis=1)

    def feature_to_sequence_analysis_df(self) -> pd.DataFrame:
        """
        Create a dataframe showing correlations between each embedding dimension
        and various numerical sequences.
        
        This method now uses the new feature analysis system while maintaining
        backward compatibility with the original property calculations.

        Returns:
            DataFrame with columns: Dimension, Property, Correlation, P_Value
        """
        # Get the numbers (tokens) from the dataframe
        numbers = self.embeddings_df["token"].astype(int).values

        properties = make_sequences(max_token=numbers.max() + 1)

        # Get embedding columns
        embedding_cols = [
            col for col in self.embeddings_df.columns if col.startswith("embeddings_")
        ]

        # Calculate correlations for each dimension and property
        correlation_data = []

        for dim_col in embedding_cols:
            dimension_values = self.embeddings_df[dim_col].values
            dimension_idx = int(dim_col.split("_")[1])  # Extract dimension index

            for prop_name, prop_values in properties.items():
                try:
                    correlation, p_value = pearsonr(dimension_values, prop_values)
                    correlation_data.append(
                        {
                            "Dimension": dimension_idx,
                            "Property": prop_name,
                            "Correlation": correlation,
                            "P_Value": p_value,
                            "Abs_Correlation": abs(correlation),
                        }
                    )
                except Exception:
                    # Handle cases where correlation can't be computed
                    correlation_data.append(
                        {
                            "Dimension": dimension_idx,
                            "Property": prop_name,
                            "Correlation": np.nan,
                            "P_Value": np.nan,
                            "Abs_Correlation": np.nan,
                        }
                    )

        df = pd.DataFrame(correlation_data)

        # Sort by absolute correlation value (descending)
        df = df.sort_values("Abs_Correlation", ascending=False, na_position="last")
        df = df.reset_index(drop=True)

        return df
    
    def dimension_sequence_correlations_df(self, sequences=None, encodings=None, **kwargs) -> pd.DataFrame:
        """
        Create a dataframe showing correlations between embedding dimensions and mathematical sequences.
        
        This method uses the new feature analysis system for more flexible sequence analysis.
        
        Args:
            sequences: List of sequence names to analyze (default: all available)
            encodings: List of encoding types ["binary", "smooth", "direct"] (default: all)
            **kwargs: Additional parameters for encodings (e.g., sigma for smooth)
            
        Returns:
            DataFrame with columns: Dimension, Property, Correlation, P_Value, Abs_Correlation
        """
        return self.feature_analysis.analyze_correlations(
            sequences=sequences, encodings=encodings, **kwargs
        )

    def get_top_dimension_property_correlations(
        self, top_n: int = 10, min_abs_correlation: float = 0.1
    ) -> pd.DataFrame:
        """
        Get the top correlations between embedding dimensions and numerical properties.

        Args:
            top_n: Number of top correlations to return
            min_abs_correlation: Minimum absolute correlation threshold

        Returns:
            DataFrame with top correlations, filtered and sorted
        """
        corr_df = self.feature_to_sequence_analysis_df()

        # Filter by minimum correlation and take top N
        filtered_df = corr_df[corr_df["Abs_Correlation"] >= min_abs_correlation].head(
            top_n
        )

        return filtered_df[["Dimension", "Property", "Correlation", "P_Value"]]

    def top_correlations_df(
        self, n_vectors: int = 20, min_correlation=0.01
    ) -> pd.DataFrame:
        """Calculate correlations between embedding dimensions."""
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
        corr_df = corr_df[corr_df["Correlation"].abs() >= min_correlation]
        corr_df.sort_values(
            by="Correlation", ascending=False, inplace=True, key=lambda x: x.abs()
        )
        corr_df.reset_index(drop=True, inplace=True)

        return corr_df

    def analyze_correlations_with_properties(self, n_vectors: int = 20):
        """Analyze correlations between embedding dimensions and numerical properties."""
        df = self.embeddings_df_justdata
        n_vectors = min(n_vectors, df.shape[1])

        # Calculate Pearson correlation coefficients
        correlations = []
        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                corr, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
                correlations.append(
                    {"Component1": i, "Component2": j, "Correlation": corr}
                )

        return pd.DataFrame(correlations)