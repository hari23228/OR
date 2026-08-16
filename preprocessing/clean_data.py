from typing import Dict, List, Any

def clean_and_validate_records(models_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    cleaned_data = {}
    valid_query_ids = None

    for model_name, info in models_data.items():
        records = info['records']
        valid_model_records = []
        model_query_ids = set()

        for rec in records:
            q_id = rec.get('index')
            query_text = rec.get('origin_query') or rec.get('prompt')
            score = rec.get('score', 0.0)
            cost = rec.get('cost', 0.0)
            p_tokens = rec.get('prompt_tokens', 0)
            c_tokens = rec.get('completion_tokens', 0)

            if q_id is None or query_text is None or str(query_text).strip() == '':
                continue
            
            # Ensure non-negative cost and valid score
            cost = max(0.0, float(cost))
            score = float(score) if score is not None else 0.0

            valid_rec = {
                'query_id': int(q_id),
                'query': str(query_text).strip(),
                'model': model_name,
                'score': score,
                'cost': cost,
                'prompt_tokens': int(p_tokens),
                'completion_tokens': int(c_tokens),
                'total_tokens': int(p_tokens) + int(c_tokens),
                'prediction': rec.get('prediction', ''),
                'ground_truth': rec.get('ground_truth', '')
            }
            valid_model_records.append(valid_rec)
            model_query_ids.add(int(q_id))

        if valid_query_ids is None:
            valid_query_ids = model_query_ids
        else:
            valid_query_ids = valid_query_ids.intersection(model_query_ids)

        cleaned_data[model_name] = valid_model_records

    # Filter to ensure complete alignment across all models
    aligned_data = {}
    for model_name, records in cleaned_data.items():
        aligned_records = [r for r in records if r['query_id'] in valid_query_ids]
        aligned_records.sort(key=lambda x: x['query_id'])
        aligned_data[model_name] = aligned_records

    print(f"Data cleaning complete. Found {len(valid_query_ids)} common aligned queries across all {len(models_data)} models.")
    return aligned_data
