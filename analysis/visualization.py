import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

COLOR_PALETTE = {
    'Gemini 2.5 Flash': '#3B82F6',   # Bright Blue
    'DeepSeek V3': '#10B981',        # Emerald Green
    'Gemini 2.5 Pro': '#8B5CF6',      # Purple
    'Claude Sonnet 4': '#F59E0B',    # Amber
    'GPT-5': '#EF4444'              # Coral Red
}

def plot_quality_by_model(stats_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        stats_df,
        x='Model',
        y='Accuracy (%)',
        text_auto='.1f',
        color='Model',
        color_discrete_map=COLOR_PALETTE,
        title='Average Quality (Accuracy %) by Model',
        labels={'Accuracy (%)': 'Accuracy (%)', 'Model': 'Language Model'}
    )
    fig.update_layout(
        template='plotly_dark',
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=13)
    )
    return fig

def plot_cost_by_model(stats_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        stats_df,
        x='Model',
        y='Total Cost ($)',
        text_auto='.4f',
        color='Model',
        color_discrete_map=COLOR_PALETTE,
        title='Total Benchmark Cost ($) by Model (3,000 Queries)',
        labels={'Total Cost ($)': 'Total Cost ($)', 'Model': 'Language Model'}
    )
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=13)
    )
    return fig

def plot_token_usage_by_model(stats_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats_df['Model'],
        y=stats_df['Avg Prompt Tokens'],
        name='Prompt Tokens',
        marker_color='#60A5FA'
    ))
    fig.add_trace(go.Bar(
        x=stats_df['Model'],
        y=stats_df['Avg Completion Tokens'],
        name='Completion Tokens',
        marker_color='#A78BFA'
    ))
    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        title='Average Token Usage per Query by Model',
        xaxis_title='Model',
        yaxis_title='Average Tokens per Query',
        font=dict(family="Inter, sans-serif", size=13)
    )
    return fig

def plot_quality_vs_cost(stats_df: pd.DataFrame, extra_points: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = go.Figure()

    # Individual models
    for _, row in stats_df.iterrows():
        model_name = row['Model']
        fig.add_trace(go.Scatter(
            x=[row['Total Cost ($)']],
            y=[row['Accuracy (%)']],
            mode='markers+text',
            name=model_name,
            text=[model_name],
            textposition="top center",
            marker=dict(size=14, color=COLOR_PALETTE.get(model_name, '#9CA3AF'))
        ))

    # Extra points (e.g., Baselines and Router points)
    if extra_points is not None and not extra_points.empty:
        for _, row in extra_points.iterrows():
            name = row.get('Method', 'Point')
            fig.add_trace(go.Scatter(
                x=[row['Total Cost ($)']],
                y=[row['Accuracy (%)']],
                mode='markers+text',
                name=name,
                text=[f"<b>{name}</b>"],
                textposition="top center",
                marker=dict(size=18, symbol='diamond', color='#F43F5E')
            ))

    fig.update_layout(
        template='plotly_dark',
        title='Quality vs Cost Trade-off Comparison',
        xaxis_title='Total Cost ($)',
        yaxis_title='Accuracy (%)',
        font=dict(family="Inter, sans-serif", size=13),
        showlegend=True
    )
    return fig
