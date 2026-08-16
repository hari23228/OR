import os
import json
import glob
from typing import Dict, List, Any, Tuple

MODEL_NAME_MAPPING = {
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'deepseek-v3-0324': 'DeepSeek V3',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'claude-sonnet-4': 'Claude Sonnet 4',
    'gpt-5': 'GPT-5'
}

def get_raw_data_dir(base_dir: str = None) -> str:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary_dir = os.path.join(base_dir, 'Data')
    if os.path.exists(primary_dir) and len(glob.glob(os.path.join(primary_dir, '*.json'))) > 0:
        return primary_dir
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    if os.path.exists(raw_dir) and len(glob.glob(os.path.join(raw_dir, '*.json'))) > 0:
        return raw_dir
    return primary_dir

def load_all_raw_models(data_dir: str = None) -> Dict[str, Dict[str, Any]]:
    if data_dir is None:
        data_dir = get_raw_data_dir()
    
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON benchmark files found in directory: {data_dir}")
    
    models_data = {}
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_model_name = data.get('model_name', os.path.basename(filepath))
            clean_name = MODEL_NAME_MAPPING.get(raw_model_name, raw_model_name)
            models_data[clean_name] = {
                'file_path': filepath,
                'metadata': {
                    'raw_model_name': raw_model_name,
                    'clean_model_name': clean_name,
                    'performance': data.get('performance'),
                    'time_taken': data.get('time_taken'),
                    'prompt_tokens': data.get('prompt_tokens'),
                    'completion_tokens': data.get('completion_tokens'),
                    'cost': data.get('cost'),
                    'counts': data.get('counts'),
                    'dataset_name': data.get('dataset_name'),
                    'split': data.get('split')
                },
                'records': data.get('records', [])
            }
    return models_data

if __name__ == '__main__':
    data = load_all_raw_models()
    print(f"Successfully loaded {len(data)} models:")
    for name, info in data.items():
        print(f" - {name}: {len(info['records'])} records, overall accuracy: {info['metadata']['performance']:.4f}, total cost: ${info['metadata']['cost']:.4f}")
