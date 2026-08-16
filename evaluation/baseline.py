import pandas as pd
from typing import Dict, Any, Tuple

def evaluate_cheapest_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # DeepSeek V3 is the cheapest model
    cheapest_model = 'DeepSeek V3'
    base_df = df[df['model'] == cheapest_model].copy()
    
    total_cost = base_df['cost'].sum()
    avg_score = base_df['score'].mean()
    N = len(base_df)
    
    summary = {
        'method': 'Baseline 1: Cheapest Model (DeepSeek V3)',
        'model_used': cheapest_model,
        'total_queries': N,
        'total_cost': total_cost,
        'average_cost': total_cost / N,
        'average_quality': avg_score,
        'accuracy_pct': avg_score * 100.0,
        'total_tokens': base_df['total_tokens'].sum(),
        'avg_tokens': base_df['total_tokens'].mean()
    }
    return base_df, summary

def evaluate_strongest_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # GPT-5 has highest benchmark score (87.37%)
    strongest_model = 'GPT-5'
    base_df = df[df['model'] == strongest_model].copy()
    
    total_cost = base_df['cost'].sum()
    avg_score = base_df['score'].mean()
    N = len(base_df)
    
    summary = {
        'method': 'Baseline 2: Strongest Model (GPT-5)',
        'model_used': strongest_model,
        'total_queries': N,
        'total_cost': total_cost,
        'average_cost': total_cost / N,
        'average_quality': avg_score,
        'accuracy_pct': avg_score * 100.0,
        'total_tokens': base_df['total_tokens'].sum(),
        'avg_tokens': base_df['total_tokens'].mean()
    }
    return base_df, summary
