import altair as alt
import pandas as pd
import numpy as np
from embanalysis.number_utils import fibonacci_proximity

class FibonacciAnalyzer:
    def __init__(self, embeddings_df):
        self.embeddings_df = embeddings_df
        self._prepare_data()
    
    def _prepare_data(self):
        """Add computed columns to the dataframe"""
        # Extract numbers from tokens (assuming tokens are numeric strings)
        self.embeddings_df = self.embeddings_df.copy()
        self.embeddings_df['number'] = self.embeddings_df['token'].astype(int)
        
        # Compute Fibonacci proximity
        numbers = self.embeddings_df['number'].values
        self.embeddings_df['fibonacci_proximity'] = fibonacci_proximity(numbers)
        
        # Compute digit count
        self.embeddings_df['digit_count'] = self.embeddings_df['number'].astype(str).str.len()
        
        # Mark actual Fibonacci numbers
        fibonacci_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        self.embeddings_df['is_fibonacci'] = self.embeddings_df['number'].isin(fibonacci_sequence)
    
    def plot_fibonacci_detector(self):
        """Visualize how dimension 3085 responds to Fibonacci proximity"""
        
        # Main scatter plot with trend line
        base = alt.Chart(self.embeddings_df).add_selection(
            alt.selection_interval(bind='scales')
        )
        
        scatter = base.mark_circle(size=30, opacity=0.6).encode(
            x=alt.X('fibonacci_proximity:Q', 
                   title='Fibonacci Proximity Score',
                   scale=alt.Scale(nice=True)),
            y=alt.Y('embeddings_3085:Q', 
                   title='Dimension 3085 Activation',
                   scale=alt.Scale(nice=True)),
            color=alt.Color('digit_count:O', 
                           title='Number of Digits',
                           scale=alt.Scale(scheme='viridis')),
            tooltip=['number:Q', 'fibonacci_proximity:Q', 'embeddings_3085:Q', 'digit_count:O']
        ).properties(
            width=400,
            height=300,
            title='Fibonacci Detector: Dimension 3085'
        )
        
        # Add regression line
        line = base.mark_line(color='red', size=3).transform_regression(
            'fibonacci_proximity', 'embeddings_3085'
        ).encode(
            x='fibonacci_proximity:Q',
            y='embeddings_3085:Q'
        )
        
        return (scatter + line).resolve_scale(color='independent')
    
    def plot_fibonacci_distribution_comparison(self):
        """Compare dimension 3085 distributions for high vs low Fibonacci proximity"""
        
        # Create high/low Fibonacci groups
        median_fib = self.embeddings_df['fibonacci_proximity'].median()
        df_labeled = self.embeddings_df.copy()
        df_labeled['fib_group'] = df_labeled['fibonacci_proximity'].apply(
            lambda x: 'High Fib Proximity' if x > median_fib else 'Low Fib Proximity'
        )
        
        # Histogram comparison
        hist = alt.Chart(df_labeled).mark_bar(opacity=0.7).encode(
            alt.X('embeddings_3085:Q', bin=alt.Bin(maxbins=30), title='Dimension 3085 Activation'),
            alt.Y('count()', title='Count'),
            alt.Color('fib_group:N', 
                     title='Fibonacci Proximity Group',
                     scale=alt.Scale(range=['#ff7f0e', '#2ca02c']))
        ).properties(
            width=400,
            height=250,
            title='Distribution: High vs Low Fibonacci Proximity'
        )
        
        return hist
    
    def plot_fibonacci_binned_trend(self):
        """Show smoothed trend using binned averages"""
        
        # Create bins for Fibonacci proximity
        df_binned = self.embeddings_df.copy()
        df_binned['fib_bin'] = pd.cut(df_binned['fibonacci_proximity'], bins=15, labels=False)
        
        # Calculate bin statistics
        bin_stats = df_binned.groupby('fib_bin').agg({
            'fibonacci_proximity': 'mean',
            'embeddings_3085': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        bin_stats.columns = ['fib_bin', 'fib_prox_mean', 'dim_3085_mean', 'dim_3085_std', 'count']
        bin_stats['error_lower'] = bin_stats['dim_3085_mean'] - bin_stats['dim_3085_std']
        bin_stats['error_upper'] = bin_stats['dim_3085_mean'] + bin_stats['dim_3085_std']
        
        # Main trend line
        line = alt.Chart(bin_stats).mark_line(
            point=True, size=3, color='blue'
        ).encode(
            x=alt.X('fib_prox_mean:Q', title='Fibonacci Proximity (binned)'),
            y=alt.Y('dim_3085_mean:Q', title='Mean Dimension 3085'),
            tooltip=['fib_prox_mean:Q', 'dim_3085_mean:Q', 'count:Q']
        )
        
        # Error bars
        error_bars = alt.Chart(bin_stats).mark_errorbar(color='blue', opacity=0.6).encode(
            x='fib_prox_mean:Q',
            y='error_lower:Q',
            y2='error_upper:Q'
        )
        
        return (line + error_bars).properties(
            width=400,
            height=250,
            title='Smoothed Fibonacci Trend (with std dev)'
        )
    
    def plot_actual_fibonacci_numbers(self):
        """Highlight how dimension 3085 responds to actual Fibonacci numbers"""
        
        # Base chart
        base = alt.Chart(self.embeddings_df)
        
        # Non-Fibonacci numbers (background)
        background = base.mark_circle(size=20, opacity=0.4, color='lightgray').encode(
            x=alt.X('number:Q', title='Number Value', scale=alt.Scale(type='log')),
            y=alt.Y('embeddings_3085:Q', title='Dimension 3085 Activation')
        ).transform_filter(
            alt.datum.is_fibonacci == False
        )
        
        # Fibonacci numbers (highlighted)
        fibonacci_points = base.mark_circle(size=100, opacity=0.8, color='red').encode(
            x='number:Q',
            y='embeddings_3085:Q',
            tooltip=['number:Q', 'embeddings_3085:Q', 'fibonacci_proximity:Q']
        ).transform_filter(
            alt.datum.is_fibonacci == True
        )
        
        # Text labels for Fibonacci numbers
        fibonacci_labels = base.mark_text(
            dx=5, dy=-5, fontSize=10, fontWeight='bold'
        ).encode(
            x='number:Q',
            y='embeddings_3085:Q',
            text='number:Q'
        ).transform_filter(
            alt.datum.is_fibonacci == True
        )
        
        return (background + fibonacci_points + fibonacci_labels).properties(
            width=500,
            height=300,
            title='Dimension 3085 Response to Actual Fibonacci Numbers'
        )
    
    def plot_fibonacci_heatmap(self):
        """Show 2D heatmap of number vs dimension 3085 activation"""
        
        # Create number ranges for binning
        df_binned = self.embeddings_df.copy()
        df_binned['number_bin'] = pd.cut(df_binned['number'], bins=20, labels=False)
        df_binned['dim_3085_bin'] = pd.cut(df_binned['embeddings_3085'], bins=15, labels=False)
        
        # Count occurrences in each bin
        heatmap_data = df_binned.groupby(['number_bin', 'dim_3085_bin']).size().reset_index(name='count')
        
        return alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('number_bin:O', title='Number Range (binned)'),
            y=alt.Y('dim_3085_bin:O', title='Dimension 3085 Range (binned)'),
            color=alt.Color('count:Q', 
                           title='Count',
                           scale=alt.Scale(scheme='blues'))
        ).properties(
            width=400,
            height=300,
            title='Density Heatmap: Numbers vs Dimension 3085'
        )
    
    def create_fibonacci_dashboard(self):
        """Combine all visualizations into a dashboard"""
        
        # Top row: main scatter + distribution
        top_row = alt.hconcat(
            self.plot_fibonacci_detector(),
            self.plot_fibonacci_distribution_comparison()
        )
        
        # Middle row: binned trend + actual Fibonacci
        middle_row = alt.hconcat(
            self.plot_fibonacci_binned_trend(),
            self.plot_actual_fibonacci_numbers()
        )
        
        # Bottom: heatmap
        bottom_row = self.plot_fibonacci_heatmap()
        
        dashboard = alt.vconcat(
            top_row,
            middle_row,
            bottom_row
        ).resolve_scale(
            color='independent'
        ).properties(
            title=alt.TitleParams(
                text='Dimension 3085: The Fibonacci Detector Dashboard',
                fontSize=16,
                fontWeight='bold'
            )
        )
        
        return dashboard

# Usage in your class:
# analyzer = FibonacciAnalyzer(embeddings_df)
# 
# # Individual plots:
# analyzer.plot_fibonacci_detector().show()
# analyzer.plot_actual_fibonacci_numbers().show()
# 
# # Full dashboard:
# analyzer.create_fibonacci_dashboard().show()


# import altair as alt
# import pandas as pd
# import numpy as np

# class FibonacciAnalyzer:
#     def __init__(self, embeddings_df):
#         self.embeddings_df = embeddings_df
    
#     def plot_fibonacci_detector(self):
#         """Visualize how dimension 3085 responds to Fibonacci proximity"""
        
#         # Main scatter plot with trend line
#         base = alt.Chart(self.embeddings_df).add_selection(
#             alt.selection_interval(bind='scales')
#         )
        
#         scatter = base.mark_circle(size=30, opacity=0.6).encode(
#             x=alt.X('fibonacci_proximity:Q', 
#                    title='Fibonacci Proximity Score',
#                    scale=alt.Scale(nice=True)),
#             y=alt.Y('embeddings_3085:Q', 
#                    title='Dimension 3085 Activation',
#                    scale=alt.Scale(nice=True)),
#             color=alt.Color('digit_count:O', 
#                            title='Number of Digits',
#                            scale=alt.Scale(scheme='viridis')),
#             tooltip=['number:Q', 'fibonacci_proximity:Q', 'embeddings_3085:Q', 'digit_count:O']
#         ).properties(
#             width=400,
#             height=300,
#             title='Fibonacci Detector: Dimension 3085'
#         )
        
#         # Add regression line
#         line = base.mark_line(color='red', size=3).transform_regression(
#             'fibonacci_proximity', 'embeddings_3085'
#         ).encode(
#             x='fibonacci_proximity:Q',
#             y='embeddings_3085:Q'
#         )
        
#         return (scatter + line).resolve_scale(color='independent')
    
#     def plot_fibonacci_distribution_comparison(self):
#         """Compare dimension 3085 distributions for high vs low Fibonacci proximity"""
        
#         # Create high/low Fibonacci groups
#         median_fib = self.embeddings_df['fibonacci_proximity'].median()
#         df_labeled = self.embeddings_df.copy()
#         df_labeled['fib_group'] = df_labeled['fibonacci_proximity'].apply(
#             lambda x: 'High Fib Proximity' if x > median_fib else 'Low Fib Proximity'
#         )
        
#         # Histogram comparison
#         hist = alt.Chart(df_labeled).mark_bar(opacity=0.7).encode(
#             alt.X('embeddings_3085:Q', bin=alt.Bin(maxbins=30), title='Dimension 3085 Activation'),
#             alt.Y('count()', title='Count'),
#             alt.Color('fib_group:N', 
#                      title='Fibonacci Proximity Group',
#                      scale=alt.Scale(range=['#ff7f0e', '#2ca02c']))
#         ).properties(
#             width=400,
#             height=250,
#             title='Distribution: High vs Low Fibonacci Proximity'
#         )
        
#         return hist
    
#     def plot_fibonacci_binned_trend(self):
#         """Show smoothed trend using binned averages"""
        
#         # Create bins for Fibonacci proximity
#         df_binned = self.embeddings_df.copy()
#         df_binned['fib_bin'] = pd.cut(df_binned['fibonacci_proximity'], bins=15, labels=False)
        
#         # Calculate bin statistics
#         bin_stats = df_binned.groupby('fib_bin').agg({
#             'fibonacci_proximity': 'mean',
#             'embeddings_3085': ['mean', 'std', 'count']
#         }).reset_index()
        
#         # Flatten column names
#         bin_stats.columns = ['fib_bin', 'fib_prox_mean', 'dim_3085_mean', 'dim_3085_std', 'count']
#         bin_stats['error_lower'] = bin_stats['dim_3085_mean'] - bin_stats['dim_3085_std']
#         bin_stats['error_upper'] = bin_stats['dim_3085_mean'] + bin_stats['dim_3085_std']
        
#         # Main trend line
#         line = alt.Chart(bin_stats).mark_line(
#             point=True, size=3, color='blue'
#         ).encode(
#             x=alt.X('fib_prox_mean:Q', title='Fibonacci Proximity (binned)'),
#             y=alt.Y('dim_3085_mean:Q', title='Mean Dimension 3085'),
#             tooltip=['fib_prox_mean:Q', 'dim_3085_mean:Q', 'count:Q']
#         )
        
#         # Error bars
#         error_bars = alt.Chart(bin_stats).mark_errorbar(color='blue', opacity=0.6).encode(
#             x='fib_prox_mean:Q',
#             y='error_lower:Q',
#             y2='error_upper:Q'
#         )
        
#         return (line + error_bars).properties(
#             width=400,
#             height=250,
#             title='Smoothed Fibonacci Trend (with std dev)'
#         )
    
#     def plot_actual_fibonacci_numbers(self):
#         """Highlight how dimension 3085 responds to actual Fibonacci numbers"""
        
#         # Mark actual Fibonacci numbers
#         fibonacci_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
#         df_marked = self.embeddings_df.copy()
#         df_marked['is_fibonacci'] = df_marked['number'].isin(fibonacci_sequence)
        
#         # Base chart
#         base = alt.Chart(df_marked)
        
#         # Non-Fibonacci numbers (background)
#         background = base.mark_circle(size=20, opacity=0.4, color='lightgray').encode(
#             x=alt.X('number:Q', title='Number Value', scale=alt.Scale(type='log')),
#             y=alt.Y('embeddings_3085:Q', title='Dimension 3085 Activation')
#         ).transform_filter(
#             alt.datum.is_fibonacci == False
#         )
        
#         # Fibonacci numbers (highlighted)
#         fibonacci_points = base.mark_circle(size=100, opacity=0.8, color='red').encode(
#             x='number:Q',
#             y='embeddings_3085:Q',
#             tooltip=['number:Q', 'embeddings_3085:Q', 'fibonacci_proximity:Q']
#         ).transform_filter(
#             alt.datum.is_fibonacci == True
#         )
        
#         # Text labels for Fibonacci numbers
#         fibonacci_labels = base.mark_text(
#             dx=5, dy=-5, fontSize=10, fontWeight='bold'
#         ).encode(
#             x='number:Q',
#             y='embeddings_3085:Q',
#             text='number:Q'
#         ).transform_filter(
#             alt.datum.is_fibonacci == True
#         )
        
#         return (background + fibonacci_points + fibonacci_labels).properties(
#             width=500,
#             height=300,
#             title='Dimension 3085 Response to Actual Fibonacci Numbers'
#         )
    
#     def plot_fibonacci_heatmap(self):
#         """Show 2D heatmap of number vs dimension 3085 activation"""
        
#         # Create number ranges for binning
#         df_binned = self.embeddings_df.copy()
#         df_binned['number_bin'] = pd.cut(df_binned['number'], bins=20, labels=False)
#         df_binned['dim_3085_bin'] = pd.cut(df_binned['embeddings_3085'], bins=15, labels=False)
        
#         # Count occurrences in each bin
#         heatmap_data = df_binned.groupby(['number_bin', 'dim_3085_bin']).size().reset_index(name='count')
        
#         return alt.Chart(heatmap_data).mark_rect().encode(
#             x=alt.X('number_bin:O', title='Number Range (binned)'),
#             y=alt.Y('dim_3085_bin:O', title='Dimension 3085 Range (binned)'),
#             color=alt.Color('count:Q', 
#                            title='Count',
#                            scale=alt.Scale(scheme='blues'))
#         ).properties(
#             width=400,
#             height=300,
#             title='Density Heatmap: Numbers vs Dimension 3085'
#         )
    
#     def create_fibonacci_dashboard(self):
#         """Combine all visualizations into a dashboard"""
        
#         # Top row: main scatter + distribution
#         top_row = alt.hconcat(
#             self.plot_fibonacci_detector(),
#             self.plot_fibonacci_distribution_comparison()
#         )
        
#         # Middle row: binned trend + actual Fibonacci
#         middle_row = alt.hconcat(
#             self.plot_fibonacci_binned_trend(),
#             self.plot_actual_fibonacci_numbers()
#         )
        
#         # Bottom: heatmap
#         bottom_row = self.plot_fibonacci_heatmap()
        
#         dashboard = alt.vconcat(
#             top_row,
#             middle_row,
#             bottom_row
#         ).resolve_scale(
#             color='independent'
#         ).properties(
#             title=alt.TitleParams(
#                 text='Dimension 3085: The Fibonacci Detector Dashboard',
#                 fontSize=16,
#                 fontWeight='bold'
#             )
#         )
        
#         return dashboard