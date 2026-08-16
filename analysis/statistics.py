import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_model_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    models = df['model'].unique()
    num_queries = df['query_id'].nunique()

    for model in models:
        m_df = df[df['model'] == model]
        avg_quality = m_df['score'].mean()
        total_cost = m_df['cost'].sum()
        avg_cost = m_df['cost'].mean()
        avg_p_tokens = m_df['prompt_tokens'].mean()
        avg_c_tokens = m_df['completion_tokens'].mean()
        avg_t_tokens = m_df['total_tokens'].mean()
        total_tokens = m_df['total_tokens'].sum()
        
        stats.append({
            'Model': model,
            'Total Queries': len(m_df),
            'Average Quality': avg_quality,
            'Accuracy (%)': avg_quality * 100.0,
            'Total Cost ($)': total_cost,
            'Average Cost ($)': avg_cost,
            'Avg Prompt Tokens': avg_p_tokens,
            'Avg Completion Tokens': avg_c_tokens,
            'Avg Total Tokens': avg_t_tokens,
            'Total Tokens': total_tokens
        })

    stats_df = pd.DataFrame(stats)
    stats_df.sort_values(by='Total Cost ($)', ascending=True, inplace=True)
    return stats_df

def print_statistical_summary(df: pd.DataFrame):
    stats_df = compute_model_statistics(df)
    print("=" * 80)
    print("MODEL STATISTICAL SUMMARY")
    print("=" * 80)
    print(stats_df[['Model', 'Accuracy (%)', 'Total Cost ($)', 'Average Cost ($)', 'Avg Total Tokens']].to_string(index=False))
    print("=" * 80)

if __name__ == '__main__':
    df = pd.read_csv('data/processed/llm_routing_dataset.csv')
    print_statistical_summary(df)
