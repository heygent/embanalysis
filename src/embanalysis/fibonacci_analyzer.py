import altair as alt
import pandas as pd
import numpy as np
from sympy import isprime
from sympy.ntheory.primetest import is_square

from embanalysis.number_utils import (
    distance_to_nearest_prime, 
    fibonacci_proximity, 
    golden_ratio_resonance, 
    is_fibonacci, 
    squareness_score
)


def make_properties_df(numbers):
    return pd.DataFrame({
        'magnitude': np.log10(numbers + 1),
        'is_even': (numbers % 2 == 0).astype(int),
        'is_prime': np.array([isprime(n) for n in numbers], dtype=int),
        'prime_proximity': -distance_to_nearest_prime(numbers),
        'perfect_square': np.array([is_square(n) for n in numbers], dtype=int),
        'squareness': squareness_score(numbers),
        'is_fibonacci': np.array([is_fibonacci(n) for n in numbers], dtype=int),
        'fibonacci_proximity': fibonacci_proximity(numbers),
        'golden_ratio_resonance': golden_ratio_resonance(numbers),
        'digit_count': [len(str(n)) for n in numbers]
    })


class ComponentAnalyzer:
    """
    A generalized class to analyze relationships between number properties and embedding dimensions.
    """

    def __init__(self, embeddings_df, properties_df=None):
        """
        Initialize the analyzer.
        
        Args:
            embeddings_df: DataFrame with embeddings data
            properties_df: Optional DataFrame with precomputed number properties
        """
        self.embeddings_df = embeddings_df.copy()
        self._prepare_data(properties_df)
    
    def _prepare_data(self, properties_df=None):
        """Add computed columns to the dataframe"""
        # Extract numbers from tokens (assuming tokens are numeric strings)
        self.embeddings_df['number'] = self.embeddings_df['token'].astype(int)
        
        # Add number properties if not provided
        if properties_df is None:
            numbers = self.embeddings_df['number'].values
            properties_df = make_properties_df(numbers)
        
        # Merge properties with embeddings
        for col in properties_df.columns:
            if col not in self.embeddings_df.columns:
                self.embeddings_df[col] = properties_df[col].values
    
    def plot_property_detector(self, property_name, dimension_col, color_by='digit_count'):
        """
        Visualize how a specific dimension responds to a number property.
        
        Args:
            property_name: Name of the property column to analyze
            dimension_col: Name of the embedding dimension column
            color_by: Column to use for color encoding
        """
        # Get display names
        property_title = property_name.replace('_', ' ').title()
        dimension_title = dimension_col.replace('_', ' ').title()
        
        # Main scatter plot with trend line
        base = alt.Chart(self.embeddings_df).add_selection(
            alt.selection_interval(bind='scales')
        )
        
        scatter = base.mark_circle(size=30, opacity=0.6).encode(
            x=alt.X(f'{property_name}:Q', 
                   title=f'{property_title} Score',
                   scale=alt.Scale(nice=True)),
            y=alt.Y(f'{dimension_col}:Q', 
                   title=f'{dimension_title} Activation',
                   scale=alt.Scale(nice=True)),
            color=alt.Color(f'{color_by}:O', 
                           title=color_by.replace('_', ' ').title(),
                           scale=alt.Scale(scheme='viridis')),
            tooltip=['number:Q', f'{property_name}:Q', f'{dimension_col}:Q', f'{color_by}:O']
        ).properties(
            width=400,
            height=300,
            title=f'{property_title} Detector: {dimension_title}'
        )
        
        # Add regression line
        line = base.mark_line(color='red', size=3).transform_regression(
            property_name, dimension_col
        ).encode(
            x=f'{property_name}:Q',
            y=f'{dimension_col}:Q'
        )
        
        return (scatter + line).resolve_scale(color='independent')
    
    def plot_property_distribution_comparison(self, property_name, dimension_col):
        """
        Compare dimension distributions for high vs low property values.
        
        Args:
            property_name: Name of the property column to analyze
            dimension_col: Name of the embedding dimension column
        """
        # Create high/low property groups
        median_prop = self.embeddings_df[property_name].median()
        df_labeled = self.embeddings_df.copy()
        property_title = property_name.replace('_', ' ').title()
        
        df_labeled['prop_group'] = df_labeled[property_name].apply(
            lambda x: f'High {property_title}' if x > median_prop else f'Low {property_title}'
        )
        
        # Histogram comparison
        dimension_title = dimension_col.replace('_', ' ').title()
        hist = alt.Chart(df_labeled).mark_bar(opacity=0.7).encode(
            alt.X(f'{dimension_col}:Q', bin=alt.Bin(maxbins=30), title=f'{dimension_title} Activation'),
            alt.Y('count()', title='Count'),
            alt.Color('prop_group:N', 
                     title=f'{property_title} Group',
                     scale=alt.Scale(range=['#ff7f0e', '#2ca02c']))
        ).properties(
            width=400,
            height=250,
            title=f'Distribution: High vs Low {property_title}'
        )
        
        return hist
    
    def plot_property_binned_trend(self, property_name, dimension_col, n_bins=15):
        """
        Show smoothed trend using binned averages.
        
        Args:
            property_name: Name of the property column to analyze
            dimension_col: Name of the embedding dimension column
            n_bins: Number of bins to use
        """
        # Create bins for property
        df_binned = self.embeddings_df.copy()
        df_binned['prop_bin'] = pd.cut(df_binned[property_name], bins=n_bins, labels=False)
        
        # Calculate bin statistics
        bin_stats = df_binned.groupby('prop_bin').agg({
            property_name: 'mean',
            dimension_col: ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        bin_stats.columns = ['prop_bin', 'prop_mean', 'dim_mean', 'dim_std', 'count']
        bin_stats['error_lower'] = bin_stats['dim_mean'] - bin_stats['dim_std']
        bin_stats['error_upper'] = bin_stats['dim_mean'] + bin_stats['dim_std']
        
        property_title = property_name.replace('_', ' ').title()
        dimension_title = dimension_col.replace('_', ' ').title()
        
        # Main trend line
        line = alt.Chart(bin_stats).mark_line(
            point=True, size=3, color='blue'
        ).encode(
            x=alt.X('prop_mean:Q', title=f'{property_title} (binned)'),
            y=alt.Y('dim_mean:Q', title=f'Mean {dimension_title}'),
            tooltip=['prop_mean:Q', 'dim_mean:Q', 'count:Q']
        )
        
        # Error bars
        error_bars = alt.Chart(bin_stats).mark_errorbar(color='blue', opacity=0.6).encode(
            x='prop_mean:Q',
            y='error_lower:Q',
            y2='error_upper:Q'
        )
        
        return (line + error_bars).properties(
            width=400,
            height=250,
            title=f'Smoothed {property_title} Trend (with std dev)'
        )
    
    def plot_property_heatmap(self, property_name, dimension_col, n_bins_x=20, n_bins_y=15):
        """
        Show 2D heatmap of property vs dimension activation.
        
        Args:
            property_name: Name of the property column to analyze
            dimension_col: Name of the embedding dimension column
            n_bins_x: Number of bins for property axis
            n_bins_y: Number of bins for dimension axis
        """
        # Create bins
        df_binned = self.embeddings_df.copy()
        df_binned['prop_bin'] = pd.cut(df_binned[property_name], bins=n_bins_x, labels=False)
        df_binned['dim_bin'] = pd.cut(df_binned[dimension_col], bins=n_bins_y, labels=False)
        
        # Count occurrences in each bin
        heatmap_data = df_binned.groupby(['prop_bin', 'dim_bin']).size().reset_index(name='count')
        
        property_title = property_name.replace('_', ' ').title()
        dimension_title = dimension_col.replace('_', ' ').title()
        
        return alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('prop_bin:O', title=f'{property_title} Range (binned)'),
            y=alt.Y('dim_bin:O', title=f'{dimension_title} Range (binned)'),
            color=alt.Color('count:Q', 
                           title='Count',
                           scale=alt.Scale(scheme='blues'))
        ).properties(
            width=400,
            height=300,
            title=f'Density Heatmap: {property_title} vs {dimension_title}'
        )
    
    def create_property_dashboard(self, property_name, dimension_col, color_by='digit_count'):
        """
        Combine all visualizations into a dashboard for a specific property-dimension pair.
        
        Args:
            property_name: Name of the property column to analyze
            dimension_col: Name of the embedding dimension column
            color_by: Column to use for color encoding in scatter plot
        """
        property_title = property_name.replace('_', ' ').title()
        dimension_title = dimension_col.replace('_', ' ').title()
        
        # Top row: main scatter + distribution
        top_row = alt.hconcat(
            self.plot_property_detector(property_name, dimension_col, color_by),
            self.plot_property_distribution_comparison(property_name, dimension_col)
        )
        
        # Middle row: binned trend + heatmap
        middle_row = alt.hconcat(
            self.plot_property_binned_trend(property_name, dimension_col),
            self.plot_property_heatmap(property_name, dimension_col)
        )
        
        dashboard = alt.vconcat(
            top_row,
            middle_row
        ).resolve_scale(
            color='independent'
        ).properties(
            title=alt.TitleParams(
                text=f'{dimension_title}: The {property_title} Detector Dashboard',
                fontSize=16,
                fontWeight='bold'
            )
        )
        
        return dashboard


# Usage Examples:
# 
# # General component analysis for any property-dimension pair:
# analyzer = ComponentAnalyzer(embeddings_df)
# 
# # Analyze any property against any dimension:
# chart = analyzer.plot_property_detector('fibonacci_proximity', 'embeddings_3085')
# chart = analyzer.plot_property_detector('prime_proximity', 'embeddings_1234')
# chart = analyzer.plot_property_detector('squareness', 'embeddings_5678')
# 
# # Create full dashboard for any property:
# dashboard = analyzer.create_property_dashboard('golden_ratio_resonance', 'embeddings_3085')
# 
# # Specialized Fibonacci analysis:
# fib_analyzer = FibonacciAnalyzer(embeddings_df)
# fib_dashboard = fib_analyzer.create_fibonacci_dashboard('embeddings_3085')
# actual_fib_chart = fib_analyzer.plot_actual_fibonacci_numbers('embeddings_3085')
# sequence_chart = fib_analyzer.plot_fibonacci_sequence_response('embeddings_3085')

class FibonacciAnalyzer(ComponentAnalyzer):
    """
    Specialized analyzer for Fibonacci-related patterns in embeddings.
    This is a simplified example of how to use ComponentAnalyzer for specific properties.
    """
    
    def __init__(self, embeddings_df):
        """Initialize with fibonacci-specific preparation."""
        super().__init__(embeddings_df)
        self._add_fibonacci_specific_data()
    
    def _add_fibonacci_specific_data(self):
        """Add Fibonacci-specific columns"""
        # Mark actual Fibonacci numbers (extended sequence)
        fibonacci_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
        self.embeddings_df['is_fibonacci'] = self.embeddings_df['number'].isin(fibonacci_sequence)
    
    def plot_fibonacci_detector(self, dimension_col):
        """Convenience method for fibonacci proximity detection."""
        return self.plot_property_detector('fibonacci_proximity', dimension_col, 'digit_count')
    
    def plot_actual_fibonacci_numbers(self, dimension_col):
        """Highlight how a dimension responds to actual Fibonacci numbers"""
        
        # Base chart
        base = alt.Chart(self.embeddings_df)
        
        # Non-Fibonacci numbers (background) - use linear scale for better spread
        background = base.mark_circle(size=15, opacity=0.3, color='lightgray').encode(
            x=alt.X('number:Q', title='Number Value', scale=alt.Scale(nice=True)),
            y=alt.Y(f'{dimension_col}:Q', title=f'{dimension_col.replace("_", " ").title()} Activation')
        ).transform_filter(
            ~alt.datum.is_fibonacci
        )
        
        # Fibonacci numbers (highlighted with different sizes)
        fibonacci_points = base.mark_circle(size=80, opacity=0.9, color='red', stroke='darkred', strokeWidth=2).encode(
            x='number:Q',
            y=f'{dimension_col}:Q',
            tooltip=['number:Q', f'{dimension_col}:Q', 'fibonacci_proximity:Q']
        ).transform_filter(
            alt.datum.is_fibonacci
        )
        
        # Text labels for Fibonacci numbers (only for smaller numbers to avoid overlap)
        fibonacci_labels = base.mark_text(
            dx=8, dy=-8, fontSize=9, fontWeight='bold', color='darkred'
        ).encode(
            x='number:Q',
            y=f'{dimension_col}:Q',
            text=alt.condition(alt.datum.number < 1000, 'number:Q', alt.value(''))
        ).transform_filter(
            alt.datum.is_fibonacci
        )
        
        # Connect Fibonacci numbers with a line to show the sequence
        fibonacci_line = base.mark_line(
            color='red', size=2, opacity=0.7, strokeDash=[5, 5]
        ).encode(
            x='number:Q',
            y=f'{dimension_col}:Q',
            order='number:Q'
        ).transform_filter(
            alt.datum.is_fibonacci
        )
        
        dimension_title = dimension_col.replace('_', ' ').title()
        return (background + fibonacci_line + fibonacci_points + fibonacci_labels).properties(
            width=600,
            height=400,
            title=f'{dimension_title} Response to Actual Fibonacci Numbers'
        )
    
    def plot_fibonacci_sequence_response(self, dimension_col):
        """Plot Fibonacci numbers by their position in the sequence"""
        
        # Filter to only Fibonacci numbers and add sequence position
        fib_data = self.embeddings_df[self.embeddings_df['is_fibonacci']].copy()
        
        # Define the Fibonacci sequence for position mapping
        fibonacci_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
        fib_positions = {fib_num: idx for idx, fib_num in enumerate(fibonacci_sequence)}
        
        # Add sequence position
        fib_data['fib_position'] = fib_data['number'].map(fib_positions)
        fib_data = fib_data.dropna(subset=['fib_position'])  # Remove any unmapped values
        
        if len(fib_data) == 0:
            # Return empty chart if no Fibonacci numbers found
            return alt.Chart().mark_text(text='No Fibonacci numbers found in data')
        
        base = alt.Chart(fib_data)
        
        # Line connecting the sequence
        line = base.mark_line(color='blue', size=3).encode(
            x=alt.X('fib_position:O', title='Fibonacci Sequence Position'),
            y=alt.Y(f'{dimension_col}:Q', title=f'{dimension_col.replace("_", " ").title()} Activation')
        )
        
        # Points for each Fibonacci number
        points = base.mark_circle(size=100, color='red', opacity=0.8).encode(
            x='fib_position:O',
            y=f'{dimension_col}:Q',
            tooltip=['number:Q', 'fib_position:Q', f'{dimension_col}:Q', 'fibonacci_proximity:Q']
        )
        
        # Labels showing the actual Fibonacci numbers
        labels = base.mark_text(dy=-15, fontSize=10, fontWeight='bold').encode(
            x='fib_position:O',
            y=f'{dimension_col}:Q',
            text='number:Q'
        )
        
        dimension_title = dimension_col.replace('_', ' ').title()
        return (line + points + labels).properties(
            width=600,
            height=300,
            title=f'{dimension_title} Response by Fibonacci Sequence Position'
        )
    
    def create_fibonacci_dashboard(self, dimension_col):
        """Create a specialized dashboard for Fibonacci analysis."""
        dimension_title = dimension_col.replace('_', ' ').title()
        
        # Top row: main scatter + distribution
        top_row = alt.hconcat(
            self.plot_fibonacci_detector(dimension_col),
            self.plot_property_distribution_comparison('fibonacci_proximity', dimension_col)
        )
        
        # Middle row: binned trend + actual Fibonacci by value
        middle_row = alt.hconcat(
            self.plot_property_binned_trend('fibonacci_proximity', dimension_col),
            self.plot_actual_fibonacci_numbers(dimension_col)
        )
        
        # Bottom row: sequence position plot + heatmap
        bottom_row = alt.hconcat(
            self.plot_fibonacci_sequence_response(dimension_col),
            self.plot_property_heatmap('fibonacci_proximity', dimension_col)
        )
        
        dashboard = alt.vconcat(
            top_row,
            middle_row,
            bottom_row
        ).resolve_scale(
            color='independent'
        ).properties(
            title=alt.TitleParams(
                text=f'{dimension_title}: The Fibonacci Detector Dashboard',
                fontSize=16,
                fontWeight='bold'
            )
        )
        
        return dashboard