from typing import Dict, Any

def calculate_cost_savings(baseline_cost: float, optimized_cost: float) -> float:
    if baseline_cost <= 0:
        return 0.0
    return ((baseline_cost - optimized_cost) / baseline_cost) * 100.0

def calculate_quality_retention(baseline_accuracy: float, optimized_accuracy: float) -> float:
    if baseline_accuracy <= 0:
        return 0.0
    return (optimized_accuracy / baseline_accuracy) * 100.0

def format_evaluation_summary(summary_dict: Dict[str, Any], baseline_strongest_cost: float = None) -> Dict[str, Any]:
    formatted = {
        'Method': summary_dict.get('method', 'Optimized Router'),
        'Accuracy (%)': f"{summary_dict['accuracy_pct']:.2f}%",
        'Total Cost ($)': f"${summary_dict['total_cost']:.4f}",
        'Average Cost ($)': f"${summary_dict['average_cost']:.6f}",
        'Avg Tokens': f"{summary_dict['avg_tokens']:.1f}"
    }
    if baseline_strongest_cost and baseline_strongest_cost > 0:
        savings = calculate_cost_savings(baseline_strongest_cost, summary_dict['total_cost'])
        formatted['Cost Savings vs Strongest (%)'] = f"{savings:.2f}%"
    return formatted
