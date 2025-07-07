from dataclasses import dataclass
from functools import cached_property
from typing import Self
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from sklearn.base import BaseEstimator

from embanalysis.feature_analysis import make_encoded_sequences

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
    def embeddings_df_without_token_data(self) -> pd.DataFrame:
        """Get just the data columns (exclude token_id and token)."""
        return self.embeddings_df.drop(["token_id", "token"], axis=1)

    def feature_to_sequence_analysis_df(self, fourier_encoding=False, compute_mutual_info=False) -> pd.DataFrame:
        """
        Create a dataframe showing correlations and mutual information between each 
        embedding dimension and various numerical sequences.
        
        This method now uses the new feature analysis system while maintaining
        backward compatibility with the original property calculations.

        Returns:
            DataFrame with columns: Dimension, Property, Encoding, Correlation, P_Value, Mutual_Info
        """
        # Get the numbers (tokens) from the dataframe

        sequences = make_encoded_sequences(len(self.embeddings_df))

        # Get embedding columns
        embedding_cols = [
            col for col in self.embeddings_df.columns if col.startswith("embeddings_")
        ]

        # Calculate correlations and mutual information for each dimension and property
        correlation_data = []

        for dim_col in embedding_cols:
            dimension_values = self.embeddings_df[dim_col].values
            dimension_idx = int(dim_col.split("_")[1])  # Extract dimension index

            for (sequence_name, encoding), prop_values in sequences.items():
                # Calculate Pearson correlation
                correlation, p_value = pearsonr(dimension_values, prop_values)

                row = {
                    "Dimension": dimension_idx,
                    "Property": sequence_name,
                    "Encoding": encoding,
                    "Correlation": correlation,
                    "P_Value": p_value,
                }

                if compute_mutual_info:
                    # Reshape for sklearn (expects 2D array for X)
                    X = dimension_values.reshape(-1, 1)
                    y = prop_values
                    mutual_info = mutual_info_regression(X, y, random_state=42)[0]
                    row["Mutual_Info"] = mutual_info

                correlation_data.append(row)

        df = pd.DataFrame(correlation_data)

        # Sort by absolute correlation value (descending)
        df = df.sort_values("Correlation", ascending=False, na_position="last", key=lambda x: x.abs())
        df = df.reset_index(drop=True)

        return df
    
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
        filtered_df = corr_df[corr_df["Correlation"].abs() >= min_abs_correlation].head(
            top_n
        )

        return filtered_df[["Dimension", "Property", "Correlation", "P_Value", "Mutual_Info"]]
    
    @cached_property
    def fourier_df(self) -> pd.DataFrame:
        """
        Perform FFT analysis on each embedding dimension and return a linear DataFrame.
        
        Returns:
            DataFrame with columns:
            - dimension: embedding dimension index
            - frequency_index: index of the frequency component
            - frequency: FFT frequencies
            - magnitude: FFT magnitude
            - phase: FFT phase
        """
        # Get embedding columns
        embedding_cols = [
            col for col in self.embeddings_df.columns if col.startswith("embeddings_")
        ]
        
        fft_data = []
        
        for dim_col in embedding_cols:
            dimension_values = self.embeddings_df[dim_col].values
            dimension_idx = int(dim_col.split("_")[1])  # Extract dimension index
            
            # Perform FFT
            fft_result = np.fft.fft(dimension_values)
            frequencies = np.fft.fftfreq(len(dimension_values))
            
            # Create DataFrame for this dimension
            for freq_idx, (freq, fft_val) in enumerate(zip(frequencies, fft_result)):
                fft_data.append({
                    'dimension': dimension_idx,
                    'frequency_index': freq_idx,
                    'frequency': freq,
                    'magnitude': np.abs(fft_val),
                    'phase': np.angle(fft_val)
                })
        
        # Create DataFrame with linear structure
        df = pd.DataFrame(fft_data)
        
        return df
    
    def fourier_dimension_df(self, dimension_idx: int) -> pd.DataFrame:
        """
        Get FFT analysis for a specific embedding dimension.
        
        Args:
            dimension_idx: The index of the embedding dimension to analyze
            
        Returns:
            DataFrame with columns:
            - frequency: FFT frequencies
            - magnitude: FFT magnitude  
            - phase: FFT phase
        """
        # Filter fourier_df for the specific dimension
        dimension_data = self.fourier_df[self.fourier_df['dimension'] == dimension_idx].copy()
        
        # Create DataFrame with the requested structure
        result_df = pd.DataFrame({
            'frequency': dimension_data['frequency'].values,
            'magnitude': dimension_data['magnitude'].values,
            'phase': dimension_data['phase'].values
        })
        
        return result_df

    def top_correlations_df(
        self, n_vectors: int = 20, min_correlation=0.01
    ) -> pd.DataFrame:
        """Calculate correlations between embedding dimensions."""
        df = self.embeddings_df_without_token_data
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
        df = self.embeddings_df_without_token_data
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
