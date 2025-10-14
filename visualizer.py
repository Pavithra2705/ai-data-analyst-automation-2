import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import plotly.io as pio
import os

# --- Original DataVisualizer class (keep your existing one) ---
class DataVisualizer:
    def __init__(self, df):
        self.df = df
        self.color_palette = px.colors.qualitative.Set3
    
    def create_histogram(self, column, bins=30):
        """Create histogram for numerical column"""
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
        """Create scatter plot"""
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
        """Create box plot"""
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
        """Create bar chart for categorical data"""
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
        """Create correlation heatmap"""
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
        """Create time series plot"""
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


# --- NEW: EfficientVisualizer class for dashboard ---
class EfficientVisualizer:
    def __init__(self, df):
        self.df = df
        # Vibrant color palettes
        self.colors = {
            'primary': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
            'gradient': px.colors.sequential.Viridis,
            'categorical': px.colors.qualitative.Bold,
            'diverging': px.colors.diverging.RdYlBu
        }
    
    def auto_detect_columns(self):
        """Automatically detect column types"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'date': date_cols
        }
    
    def create_dashboard(self, max_charts=6):
        """
        Create comprehensive interactive dashboard with vibrant colors
        Shows multiple chart types automatically based on data
        """
        col_types = self.auto_detect_columns()
        numeric_cols = col_types['numeric'][:4]
        categorical_cols = col_types['categorical'][:2]
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in dataset!")
        
        # Determine layout based on available data
        num_charts = min(6, len(numeric_cols) + len(categorical_cols))
        if num_charts >= 6:
            rows, cols_layout = 3, 2
        elif num_charts >= 4:
            rows, cols_layout = 2, 2
        elif num_charts >= 2:
            rows, cols_layout = 1, 2
        else:
            rows, cols_layout = 1, 1
        
        # Build subplot titles dynamically
        titles = []
        
        # Create subplots
        fig = make_subplots(
            rows=rows, 
            cols=cols_layout,
            subplot_titles=titles,  # Empty for now, will add later
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        chart_positions = []
        current_row = 1
        current_col = 1
        
        # Chart 1: Histogram for first numeric column
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col,
                    marker_color=self.colors['primary'][0],
                    opacity=0.75,
                    showlegend=False,
                    nbinsx=30
                ),
                row=current_row, col=current_col
            )
            titles.append(f"<b>📊 Distribution: {col}</b>")
            chart_positions.append((current_row, current_col))
            current_col += 1
            if current_col > cols_layout:
                current_col = 1
                current_row += 1
        
        # Chart 2: Box plot for second numeric column
        if len(numeric_cols) > 1 and current_row <= rows:
            col = numeric_cols[1]
            fig.add_trace(
                go.Box(
                    y=self.df[col],
                    name=col,
                    marker_color=self.colors['primary'][1],
                    showlegend=False,
                    boxmean='sd'
                ),
                row=current_row, col=current_col
            )
            titles.append(f"<b>📦 Box Plot: {col}</b>")
            chart_positions.append((current_row, current_col))
            current_col += 1
            if current_col > cols_layout:
                current_col = 1
                current_row += 1
        
        # Chart 3: Scatter plot if we have 2+ numeric columns
        if len(numeric_cols) > 2 and current_row <= rows:
            x_col, y_col = numeric_cols[0], numeric_cols[2]
            
            # Add color dimension if categorical exists
            if len(categorical_cols) > 0:
                color_map = pd.Categorical(self.df[categorical_cols[0]]).codes
            else:
                color_map = self.df[y_col]
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[x_col],
                    y=self.df[y_col],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color_map,
                        colorscale='Viridis',
                        showscale=True,
                        line=dict(width=0.5, color='white'),
                        opacity=0.7
                    ),
                    name='Scatter',
                    showlegend=False
                ),
                row=current_row, col=current_col
            )
            titles.append(f"<b>🔵 {y_col} vs {x_col}</b>")
            chart_positions.append((current_row, current_col))
            current_col += 1
            if current_col > cols_layout:
                current_col = 1
                current_row += 1
        
        # Chart 4: Bar chart for categorical data
        if len(categorical_cols) > 0 and current_row <= rows:
            col = categorical_cols[0]
            value_counts = self.df[col].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(
                    x=value_counts.values,
                    y=[str(v) for v in value_counts.index],
                    orientation='h',
                    marker=dict(
                        color=value_counts.values,
                        colorscale='Turbo',
                        showscale=False,
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False
                ),
                row=current_row, col=current_col
            )
            titles.append(f"<b>📈 Top 10: {col}</b>")
            chart_positions.append((current_row, current_col))
            current_col += 1
            if current_col > cols_layout:
                current_col = 1
                current_row += 1
        
        # Chart 5: Line/Area chart
        if len(numeric_cols) > 3 and current_row <= rows:
            col = numeric_cols[3]
            fig.add_trace(
                go.Scatter(
                    y=self.df[col].values,
                    x=list(range(len(self.df))),
                    mode='lines',
                    line=dict(color=self.colors['primary'][3], width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 160, 122, 0.3)',
                    showlegend=False
                ),
                row=current_row, col=current_col
            )
            titles.append(f"<b>📉 Trend: {col}</b>")
            chart_positions.append((current_row, current_col))
            current_col += 1
            if current_col > cols_layout:
                current_col = 1
                current_row += 1
        
        # Chart 6: Violin plot
        if len(numeric_cols) > 1 and current_row <= rows:
            col = numeric_cols[1]
            fig.add_trace(
                go.Violin(
                    y=self.df[col],
                    name=col,
                    marker_color=self.colors['primary'][4],
                    showlegend=False,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=current_row, col=current_col
            )
            titles.append(f"<b>🎻 Violin: {col}</b>")
            chart_positions.append((current_row, current_col))
        
        # Update layout with vibrant styling
        fig.update_layout(
            title=dict(
                text="<b>📊 Interactive Data Dashboard</b>",
                font=dict(size=28, color='#2C3E50', family='Arial Black'),
                x=0.5,
                xanchor='center',
                y=0.98
            ),
            height=300 * rows,
            showlegend=False,
            template='plotly_white',
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='white',
            font=dict(family='Arial', size=11, color='#2C3E50'),
            margin=dict(t=80, l=60, r=60, b=60)
        )
        
        # Manually add subplot titles as annotations
        for idx, title_text in enumerate(titles):
            if idx < len(chart_positions):
                row, col = chart_positions[idx]
                
                # Calculate position for annotation
                x_pos = (col - 1) / cols_layout + 0.5 / cols_layout
                y_pos = 1 - (row - 1) / rows - 0.05 / rows
                
                fig.add_annotation(
                    text=title_text,
                    xref="paper", yref="paper",
                    x=x_pos, y=y_pos,
                    showarrow=False,
                    font=dict(size=14, color='#34495E', family='Arial'),
                    xanchor='center',
                    yanchor='bottom'
                )
        
        # Update axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E8E8E8',
            showline=True,
            linewidth=2,
            linecolor='#BDC3C7',
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E8E8E8',
            showline=True,
            linewidth=2,
            linecolor='#BDC3C7',
            zeroline=False
        )
        
        return fig
    
    def show_dashboard(self):
        """Display the dashboard (browser-based, no export needed)"""
        try:
            fig = self.create_dashboard()
            fig.show()
            return fig
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            raise