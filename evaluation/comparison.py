import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple, List
from evaluation.baseline import evaluate_cheapest_baseline, evaluate_strongest_baseline
from evaluation.metrics import calculate_cost_savings, calculate_quality_retention
from optimization.integer_programming import solve_integer_programming_routing
from optimization.goal_programming import solve_goal_programming_routing
from prediction.capability_predictor import train_capability_predictor

def run_full_method_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # 1. Baseline 1: Cheapest Model
    _, b1_sum = evaluate_cheapest_baseline(df)
    
    # 2. Baseline 2: Strongest Model
    _, b2_sum = evaluate_strongest_baseline(df)
    
    # 3. Train Capability Predictor for ILP
    _, _, cap_df = train_capability_predictor(df)
    
    # 4. Proposed Method 1: Integer Programming Router
    _, ilp_sum = solve_integer_programming_routing(df, min_quality=0.85, predicted_capabilities=cap_df)
    
    # 5. Proposed Method 2: Goal Programming Router
    _, gp_sum = solve_goal_programming_routing(df, w_cost=0.30, w_quality=0.50, w_latency=0.20)
    
    strongest_cost = b2_sum['total_cost']
    strongest_acc = b2_sum['accuracy_pct']
    
    records = [
        {
            'Method': 'Baseline 1: Cheapest Model (DeepSeek V3)',
            'Accuracy (%)': b1_sum['accuracy_pct'],
            'Total Cost ($)': b1_sum['total_cost'],
            'Average Cost ($)': b1_sum['average_cost'],
            'Avg Tokens': b1_sum['avg_tokens'],
            'Cost Savings vs Strongest (%)': calculate_cost_savings(strongest_cost, b1_sum['total_cost']),
            'Quality Retention vs Strongest (%)': calculate_quality_retention(strongest_acc, b1_sum['accuracy_pct'])
        },
        {
            'Method': 'Baseline 2: Strongest Model (GPT-5)',
            'Accuracy (%)': b2_sum['accuracy_pct'],
            'Total Cost ($)': b2_sum['total_cost'],
            'Average Cost ($)': b2_sum['average_cost'],
            'Avg Tokens': b2_sum['avg_tokens'],
            'Cost Savings vs Strongest (%)': 0.0,
            'Quality Retention vs Strongest (%)': 100.0
        },
        {
            'Method': 'Proposed Method 1: Integer Programming Router',
            'Accuracy (%)': ilp_sum['accuracy_pct'],
            'Total Cost ($)': ilp_sum['total_cost'],
            'Average Cost ($)': ilp_sum['average_cost'],
            'Avg Tokens': ilp_sum['avg_tokens'],
            'Cost Savings vs Strongest (%)': calculate_cost_savings(strongest_cost, ilp_sum['total_cost']),
            'Quality Retention vs Strongest (%)': calculate_quality_retention(strongest_acc, ilp_sum['accuracy_pct'])
        },
        {
            'Method': 'Proposed Method 2: Goal Programming Router',
            'Accuracy (%)': gp_sum['accuracy_pct'],
            'Total Cost ($)': gp_sum['total_cost'],
            'Average Cost ($)': gp_sum['average_cost'],
            'Avg Tokens': gp_sum['avg_tokens'],
            'Cost Savings vs Strongest (%)': calculate_cost_savings(strongest_cost, gp_sum['total_cost']),
            'Quality Retention vs Strongest (%)': calculate_quality_retention(strongest_acc, gp_sum['accuracy_pct'])
        }
    ]
    
    comp_df = pd.DataFrame(records)
    raw_summaries = {
        'baseline_cheapest': b1_sum,
        'baseline_strongest': b2_sum,
        'ilp_router': ilp_sum,
        'gp_router': gp_sum
    }
    return comp_df, raw_summaries

def plot_method_comparison_charts(comp_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    # Cost Comparison Chart
    fig_cost = px.bar(
        comp_df,
        x='Method',
        y='Total Cost ($)',
        text_auto='.2f',
        color='Method',
        title='Total Cost ($) Comparison across Routing Methods',
        labels={'Total Cost ($)': 'Total Cost ($)', 'Method': 'Method'}
    )
    fig_cost.update_layout(template='plotly_dark', showlegend=False, font=dict(family="Inter, sans-serif"))

    # Quality vs Cost Tradeoff Chart
    fig_tradeoff = px.scatter(
        comp_df,
        x='Total Cost ($)',
        y='Accuracy (%)',
        text='Method',
        size=[16, 16, 20, 20],
        color='Method',
        title='Quality vs Cost Trade-Off Curve'
    )
    fig_tradeoff.update_traces(textposition='top center')
    fig_tradeoff.update_layout(template='plotly_dark', font=dict(family="Inter, sans-serif"))

    return fig_cost, fig_tradeoff

if __name__ == '__main__':
    df = pd.read_csv('data/processed/llm_routing_dataset.csv')
    comp_df, summaries = run_full_method_comparison(df)
    print("\nFULL METHOD COMPARISON SUMMARY:")
    print("=" * 100)
    print(comp_df.to_string(index=False))
    print("=" * 100)
