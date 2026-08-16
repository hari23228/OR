import os
import pandas as pd
from preprocessing.load_data import load_all_raw_models
from preprocessing.clean_data import clean_and_validate_records

def merge_and_save_dataset(base_dir: str = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_models = load_all_raw_models(os.path.join(base_dir, 'Data'))
    cleaned_records_by_model = clean_and_validate_records(raw_models)

    all_rows = []
    for model_name, records in cleaned_records_by_model.items():
        all_rows.extend(records)

    df = pd.DataFrame(all_rows)
    
    output_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'llm_routing_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path} with {len(df)} total rows across {df['model'].nunique()} models.")
    return df

if __name__ == '__main__':
    merge_and_save_dataset()
