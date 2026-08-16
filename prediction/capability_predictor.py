import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from prediction.feature_extraction import extract_query_features
from typing import Dict, Any, Tuple

# Order models by cost ascending
MODEL_COST_ORDER = [
    'DeepSeek V3',        # Level 1
    'Gemini 2.5 Flash',   # Level 2
    'Claude Sonnet 4',    # Level 3
    'GPT-5',              # Level 4
    'Gemini 2.5 Pro'      # Level 5
]

MODEL_TO_LEVEL = {model: idx + 1 for idx, model in enumerate(MODEL_COST_ORDER)}
LEVEL_TO_MODEL = {idx + 1: model for idx, model in enumerate(MODEL_COST_ORDER)}

def derive_ground_truth_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    query_records = []
    
    for q_id, group in df.groupby('query_id'):
        query_text = group['query'].iloc[0]
        
        # Sort group by model cost order
        group['cost_rank'] = group['model'].map(MODEL_TO_LEVEL)
        group_sorted = group.sort_values(by='cost_rank')
        
        # Find cheapest model with score == 1.0
        correct_models = group_sorted[group_sorted['score'] == 1.0]
        if not correct_models.empty:
            min_level_model = correct_models.iloc[0]['model']
            required_level = MODEL_TO_LEVEL[min_level_model]
        else:
            required_level = 5 # Needs strongest capability
            
        query_records.append({
            'query_id': q_id,
            'query': query_text,
            'required_capability': required_level,
            'min_capable_model': LEVEL_TO_MODEL[required_level]
        })
        
    return pd.DataFrame(query_records)

def train_capability_predictor(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, Any], pd.DataFrame]:
    capabilities_df = derive_ground_truth_capabilities(df)
    queries = capabilities_df['query'].tolist()
    y = capabilities_df['required_capability'].values
    
    X, feature_names = extract_query_features(queries, max_tfidf_features=300)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimized Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred_test = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    
    # Predict for all queries to use in optimization stage
    all_preds = clf.predict(X)
    capabilities_df['predicted_capability'] = all_preds
    capabilities_df['predicted_model'] = capabilities_df['predicted_capability'].map(LEVEL_TO_MODEL)
    
    metrics = {
        'test_accuracy': acc,
        'feature_names': feature_names,
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
    }
    
    print(f"Capability Predictor trained successfully. Test Accuracy: {acc * 100.0:.2f}%")
    return clf, metrics, capabilities_df

if __name__ == '__main__':
    df = pd.read_csv('data/processed/llm_routing_dataset.csv')
    clf, metrics, cap_df = train_capability_predictor(df)
    print("\nSample Capability Predictions:")
    print(cap_df[['query_id', 'required_capability', 'predicted_capability', 'min_capable_model', 'predicted_model']].head(10))
