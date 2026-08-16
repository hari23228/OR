import pulp
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from prediction.capability_predictor import MODEL_COST_ORDER

def solve_goal_programming_routing(
    df: pd.DataFrame,
    w_cost: float = 0.30,
    w_quality: float = 0.50,
    w_latency: float = 0.20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Solves Cost-Aware LLM Routing using Multi-Objective Goal Programming (PuLP).
    
    Objective:
      Minimize sum_{i, j} (w_cost * d_cost[i,j] + w_quality * d_quality[i,j] + w_latency * d_latency[i,j]) * x[i,j]
    where d_cost, d_quality, d_latency are normalized deviations from target goals.
    """
    # Normalize weights so they sum to 1.0
    total_w = w_cost + w_quality + w_latency
    w_cost /= max(1e-6, total_w)
    w_quality /= max(1e-6, total_w)
    w_latency /= max(1e-6, total_w)

    queries = df['query_id'].unique()
    models = MODEL_COST_ORDER
    N = len(queries)

    cost_pivot = df.pivot(index='query_id', columns='model', values='cost')[models]
    score_pivot = df.pivot(index='query_id', columns='model', values='score')[models]
    token_pivot = df.pivot(index='query_id', columns='model', values='total_tokens')[models]

    min_cost_global = df['cost'].min()
    max_cost_global = df['cost'].max()
    cost_range = max(1e-6, max_cost_global - min_cost_global)

    min_token_global = df['total_tokens'].min()
    max_token_global = df['total_tokens'].max()
    token_range = max(1.0, max_token_global - min_token_global)

    prob = pulp.LpProblem("Goal_Programming_LLM_Routing", pulp.LpMinimize)

    # Decision variables
    x = {}
    penalty = {}
    for q_id in queries:
        for m in models:
            x[q_id, m] = pulp.LpVariable(f"x_{q_id}_{m}", cat=pulp.LpBinary)
            
            c = cost_pivot.loc[q_id, m]
            s = score_pivot.loc[q_id, m]
            t = token_pivot.loc[q_id, m]
            
            # Normalized deviations
            d_cost = (c - min_cost_global) / cost_range
            d_quality = 1.0 - s
            d_latency = (t - min_token_global) / token_range

            penalty[q_id, m] = w_cost * d_cost + w_quality * d_quality + w_latency * d_latency

    # Objective Function
    prob += pulp.lpSum(penalty[q_id, m] * x[q_id, m] for q_id in queries for m in models)

    # Constraint: Exactly 1 model per query
    for q_id in queries:
        prob += pulp.lpSum(x[q_id, m] for m in models) == 1, f"OneModel_{q_id}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    routing_results = []
    total_cost = 0.0
    total_score = 0.0
    total_tokens = 0

    for q_id in queries:
        selected_model = None
        for m in models:
            if pulp.value(x[q_id, m]) is not None and pulp.value(x[q_id, m]) > 0.5:
                selected_model = m
                break
        
        if selected_model is None:
            selected_model = models[0]

        c = cost_pivot.loc[q_id, selected_model]
        s = score_pivot.loc[q_id, selected_model]
        t = token_pivot.loc[q_id, selected_model]

        total_cost += c
        total_score += s
        total_tokens += t

        routing_results.append({
            'query_id': q_id,
            'selected_model': selected_model,
            'cost': c,
            'score': s,
            'tokens': t
        })

    results_df = pd.DataFrame(routing_results)

    summary = {
        'status': pulp.LpStatus[status],
        'weights': {'cost': w_cost, 'quality': w_quality, 'latency': w_latency},
        'total_queries': N,
        'total_cost': total_cost,
        'average_cost': total_cost / N,
        'average_quality': total_score / N,
        'accuracy_pct': (total_score / N) * 100.0,
        'total_tokens': total_tokens,
        'avg_tokens': total_tokens / N,
        'model_distribution': results_df['selected_model'].value_counts().to_dict()
    }

    return results_df, summary

if __name__ == '__main__':
    df = pd.read_csv('data/processed/llm_routing_dataset.csv')
    results_df, summary = solve_goal_programming_routing(df, w_cost=0.30, w_quality=0.50, w_latency=0.20)
    print("\nGoal Programming Optimization Results:")
    print(f"Status: {summary['status']}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Accuracy: {summary['accuracy_pct']:.2f}%")
    print(f"Model Distribution: {summary['model_distribution']}")
