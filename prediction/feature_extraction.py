import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any

REASONING_KEYWORDS = [
    'why', 'how', 'explain', 'calculate', 'prove', 'determine', 'find', 'derive',
    'assume', 'suppose', 'show', 'compare', 'contrast', 'evaluate', 'analyze',
    'which of the following', 'what is', 'solve', 'infer'
]

def extract_query_features(queries: List[str], max_tfidf_features: int = 300) -> Tuple[np.ndarray, List[str]]:
    feature_rows = []
    
    for q in queries:
        q_str = str(q) if q is not None else ""
        q_lower = q_str.lower()
        
        char_len = len(q_str)
        words = q_str.split()
        word_count = len(words)
        avg_word_len = char_len / max(1, word_count)
        
        # Option structure
        options_count = len(re.findall(r'[A-J]\.\s', q_str))
        opt_matches = re.findall(r'[A-J]\.\s*(.*?)(?=[A-J]\.|\Z)', q_str, re.DOTALL)
        max_opt_len = max([len(m) for m in opt_matches]) if opt_matches else 0
        avg_opt_len = float(np.mean([len(m) for m in opt_matches])) if opt_matches else 0.0
        
        # Numerical and symbol metrics
        digits_count = len(re.findall(r'\d+', q_str))
        math_count = len(re.findall(r'[\+\-\*\/\=\^\<\>\%\\\$]', q_str))
        code_count = len(re.findall(r'[\{\}\[\]\(\)\;\:]', q_str))
        upper_ratio = sum(1 for c in q_str if c.isupper()) / max(1, char_len)
        question_count = q_str.count('?')
        reasoning_score = sum(1 for kw in REASONING_KEYWORDS if kw in q_lower)
        
        # Domain indicators
        is_math = int(any(k in q_lower for k in ['matrix', 'integral', 'derivative', 'equation', 'probability', 'vector', 'prime', 'function', 'theorem']))
        is_phys = int(any(k in q_lower for k in ['force', 'energy', 'mass', 'velocity', 'quantum', 'magnetic', 'electric', 'photon', 'thermodynamic']))
        is_cs = int(any(k in q_lower for k in ['algorithm', 'complexity', 'graph', 'tree', 'binary', 'sort', 'array', 'pointer', 'recursion']))
        is_bio = int(any(k in q_lower for k in ['dna', 'protein', 'cell', 'gene', 'enzyme', 'rna', 'membrane', 'chromosome', 'organism']))
        is_econ = int(any(k in q_lower for k in ['demand', 'supply', 'market', 'inflation', 'gdp', 'monopoly', 'price', 'equilibrium']))
        
        feature_rows.append([
            char_len,
            word_count,
            avg_word_len,
            options_count,
            max_opt_len,
            avg_opt_len,
            digits_count,
            math_count,
            code_count,
            upper_ratio,
            question_count,
            reasoning_score,
            is_math,
            is_phys,
            is_cs,
            is_bio,
            is_econ
        ])

    num_feature_names = [
        'char_len', 'word_count', 'avg_word_len', 'options_count',
        'max_opt_len', 'avg_opt_len', 'digits_count', 'math_count',
        'code_count', 'upper_ratio', 'question_count', 'reasoning_score',
        'is_math', 'is_phys', 'is_cs', 'is_bio', 'is_econ'
    ]
    num_features = np.array(feature_rows, dtype=np.float32)
    scaler = StandardScaler()
    num_features_scaled = scaler.fit_transform(num_features)

    # TF-IDF Features with unigrams and bigrams
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf_features, sublinear_tf=True, stop_words='english')
    tfidf_matrix = tfidf.fit_transform([str(q) for q in queries]).toarray()
    tfidf_feature_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]

    X = np.hstack([num_features_scaled, tfidf_matrix])
    feature_names = num_feature_names + tfidf_feature_names

    return X, feature_names

