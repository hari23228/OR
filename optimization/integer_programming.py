import pulp
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from prediction.capability_predictor import MODEL_COST_ORDER, MODEL_TO_LEVEL

def solve_integer_programming_routing(
    df: pd.DataFrame,
    min_quality: float = 0.85,
    predicted_capabilities: pd.DataFrame = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Solves Cost-Aware LLM Routing using Integer Linear Programming (PuLP).
    
    Variables:
      x[i, j] = 1 if query i is assigned to model j, else 0
    Constraints:
      1. sum_j x[i, j] = 1 for all i (One model per query)
      2. Model j capability >= predicted capability level L[i] for query i
         (or Quality Q[i, j] >= min_quality)
    Objective:
      Minimize sum_{i, j} Cost[i, j] * x[i, j]
    """
    queries = df['query_id'].unique()
    models = MODEL_COST_ORDER
    N = len(queries)
    M = len(models)

    # Pivot cost and score tables
    cost_pivot = df.pivot(index='query_id', columns='model', values='cost')[models]
    score_pivot = df.pivot(index='query_id', columns='model', values='score')[models]
    token_pivot = df.pivot(index='query_id', columns='model', values='total_tokens')[models]

    prob = pulp.LpProblem("Cost_Aware_LLM_Routing_ILP", pulp.LpMinimize)

    # Decision variables
    x = {}
    for q_id in queries:
        for m in models:
            x[q_id, m] = pulp.LpVariable(f"x_{q_id}_{m}", cat=pulp.LpBinary)

    # Objective: Minimize total cost
    prob += pulp.lpSum(cost_pivot.loc[q_id, m] * x[q_id, m] for q_id in queries for m in models)

    # Constraint 1: Exactly 1 model per query
    for q_id in queries:
        prob += pulp.lpSum(x[q_id, m] for m in models) == 1, f"OneModel_{q_id}"

    # Constraint 2: Quality / Capability constraint per query
    if predicted_capabilities is not None and 'predicted_capability' in predicted_capabilities.columns:
        cap_dict = dict(zip(predicted_capabilities['query_id'], predicted_capabilities['predicted_capability']))
        for q_id in queries:
            req_cap = cap_dict.get(q_id, 1)
            # Allowed models are those with capability level >= req_cap
            allowed_models = [m for m in models if MODEL_TO_LEVEL[m] >= req_cap]
            if allowed_models:
                prob += pulp.lpSum(x[q_id, m] for m in allowed_models) == 1, f"Capability_{q_id}"
    else:
        # Fallback: model must satisfy score >= min_quality or overall quality >= min_quality
        for q_id in queries:
            prob += pulp.lpSum(score_pivot.loc[q_id, m] * x[q_id, m] for m in models) >= min_quality, f"MinQuality_{q_id}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    # Extract solution
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
            selected_model = models[0] # Default fallback

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
    from prediction.capability_predictor import train_capability_predictor
    _, _, cap_df = train_capability_predictor(df)
    results_df, summary = solve_integer_programming_routing(df, min_quality=0.85, predicted_capabilities=cap_df)
    print("\nInteger Programming Optimization Results:")
    print(f"Status: {summary['status']}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Accuracy: {summary['accuracy_pct']:.2f}%")
    print(f"Model Distribution: {summary['model_distribution']}")
