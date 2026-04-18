"""
eda_charts.py — Chart functions using Plotly
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
import numpy as np

def apply_theme(fig):
    """Applies the default dark theme to all Plotly figures."""
    fig.update_layout(
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.1)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.1)")
    return fig

def plot_distribution(df, column):
    """
    Plots a histogram with KDE for the specified column.
    """
    try:
        data = df[column].dropna()
        if len(data) == 0:
            return go.Figure().update_layout(title="Not enough data")

        # Create base figure with histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, 
            name='Histogram', 
            marker_color='#00d4ff', 
            opacity=0.7, 
            histnorm='probability density'
        ))

        # Add KDE
        kde = scipy.stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 200)
        fig.add_trace(go.Scatter(
            x=x_range, 
            y=kde(x_range), 
            mode='lines', 
            name='KDE', 
            line=dict(color='#7c3aed', width=2)
        ))

        fig.update_layout(title=f"Distribution of {column}", showlegend=False)
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_boxplot(df, column):
    """
    Plots a box plot for the specified numerical column.
    """
    try:
        data = df[column].dropna()
        fig = px.box(
            df, 
            y=column, 
            title=f"Box Plot — {column}",
            color_discrete_sequence=['#00d4ff']
        )
        # update outlier marker
        fig.update_traces(marker=dict(color='#ef4444', symbol='diamond'))
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_bar_chart(df, column):
    """
    Plots a horizontal bar chart of value counts.
    """
    try:
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        if len(counts) > 10:
            counts = counts.head(10)
        
        counts = counts.sort_values('Count', ascending=True)

        fig = px.bar(
            counts, 
            x='Count', 
            y=column, 
            orientation='h', 
            title=f"Bar Chart — {column} (Top {len(counts)})",
            color='Count',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_correlation_heatmap(df):
    """
    Plots a highly stylized annotated correlation heatmap.
    """
    try:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return go.Figure().update_layout(title="No numerical columns for correlation")
        
        corr = num_df.corr().round(2)
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_pairplot(df, columns, color_col=None):
    """
    Plots a scatter matrix of the specified columns.
    """
    try:
        kwargs = dict(
            title="Scatter Matrix (Pairplot)",
            dimensions=columns
        )
        if color_col and color_col in df.columns:
            kwargs['color'] = color_col
            
        fig = px.scatter_matrix(df, **kwargs)
        
        # Make markers smaller and slightly transparent
        fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.7))
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_missing_heatmap(df):
    """
    Visualizes missing values in the dataframe (True = missing, False = present).
    """
    try:
        missing_matrix = df.isnull()
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.T.astype(int).values,
            y=missing_matrix.columns,
            x=missing_matrix.index,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#10b981"], [1, "#ef4444"]], # 0 (False) is green, 1 (True) is red
            showscale=False
        ))
        
        fig.update_layout(title="Missing Value Heatmap")
        fig.update_yaxes(autorange="reversed")
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_skewness_kurtosis(df):
    """
    Returns subplot with a horizontal bar of skewness and kurtosis.
    """
    from plotly.subplots import make_subplots
    try:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return go.Figure().update_layout(title="No numerical columns")
            
        skewness = num_df.skew().dropna().sort_values()
        kurtosis = num_df.kurtosis().dropna()
        # Sort kurtosis in the same order as skewness for alignment
        kurtosis = kurtosis.loc[skewness.index]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Skewness", "Kurtosis"), horizontal_spacing=0.15)
        
        # Skewness colors: Orange if >1, Blue if <-1, Green if -1 to 1
        skew_colors = np.where(skewness > 1, '#f59e0b', np.where(skewness < -1, '#3b82f6', '#10b981'))

        fig.add_trace(
            go.Bar(y=skewness.index, x=skewness.values, orientation='h', marker_color=skew_colors, name="Skewness"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(y=kurtosis.index, x=kurtosis.values, orientation='h', marker_color='#7c3aed', name="Kurtosis"),
            row=1, col=2
        )

        fig.update_layout(title="Skewness & Kurtosis", showlegend=False)
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_count_plot(df, column):
    """
    Vertical bar chart with value labels on top of each bar.
    """
    try:
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        
        if len(counts) > 20:
             counts = counts.head(20) # Limit for readability
             
        fig = px.bar(
            counts, 
            x=column, 
            y='Count',
            text='Count',
            title=f"Count Plot — {column}",
            color='Count',
            color_continuous_scale=px.colors.sequential.Plotly3
        )
        fig.update_traces(textposition='outside')
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def plot_line_chart(df, date_col, value_col):
    """
    Plots a time-series line chart sorted by the date column.
    """
    try:
        data = df.sort_values(by=date_col).copy()
        
        fig = px.line(
            data, 
            x=date_col, 
            y=value_col, 
            title=f"{value_col} over {date_col}",
            markers=True
        )
        
        fig.update_traces(
            line=dict(color='#00d4ff'),
            marker=dict(symbol='diamond', color='#7c3aed', size=8)
        )
        return apply_theme(fig)
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}")

def get_insight(df, chart_type, column=None):
    """
    Returns auto-generated insight string based on chart_type.
    """
    try:
        if column and column not in df.columns:
            return ""

        if chart_type == 'distribution':
            mean = df[column].mean()
            std = df[column].std()
            skew = df[column].skew()
            direction = "right" if skew > 0.5 else "left" if skew < -0.5 else "normal"
            return f"**{column} Analysis:** Mean is {mean:.2f}, standard deviation is {std:.2f}. The data is **{direction}-skewed** (Skewness={skew:.2f})."
            
        elif chart_type == 'boxplot':
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            median = df[column].median()
            outliers = sum((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr)))
            return f"**Boxplot Insights:** Median={median:.2f}, IQR={iqr:.2f}. Count of outliers detected: **{outliers}**."
            
        elif chart_type == 'bar' or chart_type == 'count':
            top_val = df[column].value_counts().index[0]
            top_count = df[column].value_counts().values[0]
            uniques = df[column].nunique()
            return f"**Frequency Insight:** In {uniques} unique categories, **'{top_val}'** is the most frequent with {top_count} occurrences."
            
        elif chart_type == 'correlation':
            return "Positive values (blue) mean variables increase together. Negative values (red) mean one increases while the other decreases."
            
        elif chart_type == 'pairplot':
            return "Scatter matrix useful for spotting clusters or linear/non-linear relationships between multiple numerical variables at once."
            
        elif chart_type == 'missing':
            total_missing = df.isnull().sum().sum()
            return f"Overall missing values across the dataset: **{total_missing}**."

        elif chart_type == 'skewness':
            return "High skewness (>1 or <-1) indicates long tails. High kurtosis indicates heavily clustered data around the mean (peaked)."
            
        elif chart_type == 'line':
            avg_val = df[column].mean()
            max_val = df[column].max()
            return f"**Trend:** On average {column} is {avg_val:.2f}. The peak value observed over the timeline is {max_val:.2f}."
            
    except Exception as e:
        return f"Error generating insight: {e}"
        
    return "No insights available for this selection."
