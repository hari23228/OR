import pandas as pd
from typing import List, Dict, Any
from optimization.integer_programming import solve_integer_programming_routing
from optimization.goal_programming import solve_goal_programming_routing

def run_sensitivity_analysis(df: pd.DataFrame, predicted_capabilities: pd.DataFrame = None) -> pd.DataFrame:
    sensitivity_records = []

    # 1. Sensitivity to Goal Programming Weights
    weight_configs = [
        ('Balanced', 0.30, 0.50, 0.20),
        ('Cost-Focused', 0.60, 0.30, 0.10),
        ('Quality-Focused', 0.10, 0.80, 0.10),
        ('Latency-Focused', 0.20, 0.30, 0.50),
    ]

    for config_name, w_c, w_q, w_l in weight_configs:
        _, summary = solve_goal_programming_routing(df, w_cost=w_c, w_quality=w_q, w_latency=w_l)
        sensitivity_records.append({
            'Method': 'Goal Programming',
            'Configuration': config_name,
            'Quality Threshold / Weight': f"w_Q={w_q:.2f}, w_C={w_c:.2f}, w_L={w_l:.2f}",
            'Total Cost ($)': summary['total_cost'],
            'Accuracy (%)': summary['accuracy_pct'],
            'Avg Tokens': summary['avg_tokens'],
            'Primary Model': max(summary['model_distribution'], key=summary['model_distribution'].get)
        })

    # 2. Sensitivity to Minimum Quality Requirement in ILP
    quality_thresholds = [0.80, 0.85, 0.90, 0.95]
    for q_min in quality_thresholds:
        _, summary = solve_integer_programming_routing(df, min_quality=q_min, predicted_capabilities=None)
        sensitivity_records.append({
            'Method': 'Integer Programming',
            'Configuration': f"Q_min={int(q_min*100)}%",
            'Quality Threshold / Weight': f"Q_min >= {q_min:.2f}",
            'Total Cost ($)': summary['total_cost'],
            'Accuracy (%)': summary['accuracy_pct'],
            'Avg Tokens': summary['avg_tokens'],
            'Primary Model': max(summary['model_distribution'], key=summary['model_distribution'].get)
        })

    sensitivity_df = pd.DataFrame(sensitivity_records)
    return sensitivity_df

if __name__ == '__main__':
    df = pd.read_csv('data/processed/llm_routing_dataset.csv')
    sens_df = run_sensitivity_analysis(df)
    print("\nSensitivity Analysis Results:")
    print(sens_df.to_string(index=False))
