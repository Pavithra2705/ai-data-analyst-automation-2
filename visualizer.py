import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        self.color_palette = px.colors.qualitative.Set3
    
    def create_histogram(self, column, bins=30):
        """
        Create histogram for numerical column
        """
        fig = px.histogram(
            self.df, 
            x=column,
            nbins=bins,
            title=f'Distribution of {column}',
            marginal="box",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            showlegend=False,
            height=500
        )
        
        # Add statistics annotation
        mean_val = self.df[column].mean()
        median_val = self.df[column].median()
        std_val = self.df[column].std()
        
        fig.add_annotation(
            text=f"Mean: {mean_val:.2f}<br>Median: {median_val:.2f}<br>Std: {std_val:.2f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        return fig
    
    def create_scatter_plot(self, x_col, y_col, color_col=None, size_col=None):
        """
        Create scatter plot
        """
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=f'{y_col} vs {x_col}',
            trendline="ols",
            color_discrete_sequence=self.color_palette
        )
        
        # Calculate correlation
        correlation = self.df[x_col].corr(self.df[y_col])
        
        fig.add_annotation(
            text=f"Correlation: {correlation:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        return fig
    
    def create_box_plot(self, column, group_by=None):
        """
        Create box plot
        """
        if group_by:
            fig = px.box(
                self.df,
                x=group_by,
                y=column,
                title=f'Box Plot of {column} by {group_by}',
                color_discrete_sequence=self.color_palette
            )
        else:
            fig = px.box(
                self.df,
                y=column,
                title=f'Box Plot of {column}',
                color_discrete_sequence=self.color_palette
            )
        
        return fig
    
    def create_bar_chart(self, column, top_n=20):
        """
        Create bar chart for categorical data
        """
        value_counts = self.df[column].value_counts().head(top_n)
        
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            title=f'Top {len(value_counts)} values in {column}',
            labels={'x': 'Count', 'y': column},
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(height=max(400, len(value_counts) * 30))
        
        return fig
    
    def create_correlation_heatmap(self):
        """
        Create correlation heatmap
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            height=600,
            width=800
        )
        
        return fig
    
    def create_time_series(self, date_col, value_col, group_by=None):
        """
        Create time series plot
        """
        df_sorted = self.df.sort_values(date_col)
        
        if group_by:
            fig = px.line(
                df_sorted,
                x=date_col,
                y=value_col,
                color=group_by,
                title=f'{value_col} over time by {group_by}',
                color_discrete_sequence=self.color_palette
            )
        else:
            fig = px.line(
                df_sorted,
                x=date_col,
                y=value_col,
                title=f'{value_col} over time',
                color_discrete_sequence=self.color_palette
            )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text=value_col)
        
        return fig
    
    def create_distribution_comparison(self, columns):
        """
        Compare distributions of multiple columns
        """
        if len(columns) < 2:
            return None
        
        fig = make_subplots(
            rows=len(columns), cols=1,
            subplot_titles=[f'Distribution of {col}' for col in columns],
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(columns):
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, showlegend=False),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(columns),
            title_text="Distribution Comparison"
        )
        
        return fig
    
    def create_pair_plot_data(self, columns, sample_size=1000):
        """
        Prepare data for pair plot (returns data for external plotting)
        """
        if len(self.df) > sample_size:
            sample_df = self.df[columns].sample(sample_size)
        else:
            sample_df = self.df[columns]
        
        return sample_df
    
    def create_summary_dashboard(self):
        """
        Create a comprehensive summary dashboard
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4 for display
        
        if len(numerical_cols) < 1:
            return None
        
        # Create subplots
        rows = 2
        cols = 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'Distribution of {col}' for col in numerical_cols[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(numerical_cols[:4]):
            row = (i // cols) + 1
            col_pos = (i % cols) + 1
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            height=800,
            title_text="Data Summary Dashboard",
            showlegend=False
        )
        
        return fig
    
    def export_chart_as_image(self, fig, filename="chart.png"):
        """
        Export plotly figure as PNG image
        """
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            return img_bytes
        except Exception as e:
            # Fallback to HTML export
            html_str = fig.to_html()
            return html_str.encode()
