import os
import sys
import json
import numpy as np
import pandas as pd

# Mapping from numeric class indices back to sport names
INT_TO_LABEL = {0: 'Baseball', 1: 'Basketball', 2: 'Soccer'}

def load_json_safe(filepath):
    """Loads a JSON file, printing a clear warning if it is missing."""
    if not os.path.exists(filepath):
        print(f"[WARNING] Required metadata/model file not found: '{filepath}'")
        raise FileNotFoundError(f"Missing required file: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_tfidf_inference(text_series, prefix, saved_idf):
    """Applies saved IDF weights to new text data."""
    docs = text_series.fillna('null').astype(str).str.lower().str.split(r'\W+')
    docs = docs.apply(lambda x: [w for w in x if w])
    
    tfidf_matrix = []
    for doc in docs:
        doc_len = len(doc)
        doc_counts = {}
        for w in doc:
            doc_counts[w] = doc_counts.get(w, 0) + 1
            
        row = {}
        for word, idf_val in saved_idf.items():
            tf = doc_counts.get(word, 0) / doc_len if doc_len > 0 else 0
            row[f'{prefix}_{word}'] = tf * idf_val
        tfidf_matrix.append(row)
    return pd.DataFrame(tfidf_matrix, index=text_series.index)

def preprocess_for_inference(df):
    """Applies the exact training preprocessing pipeline to raw unseen data."""
    medians = load_json_safe('imputed_medians.json')
    tfidf_metadata = load_json_safe('tfidf_metadata.json')
    scaling_params = load_json_safe('scaling_params.json')
    price_cap_params = load_json_safe('ticket_price_cap.json')
        
    expected_features = list(scaling_params.keys())

    # 1. Rename columns
    column_mapping = {
        'How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy.': 'ease_of_play',
        'Success in this sport depends heavily on strategy and planning.': 'strategy_dependent',
        'Players must be comfortable with a high risk of injury.': 'injury_risk',
        'Individual brilliance can outweigh teamwork in this sport.': 'individual_brilliance',
        'Where might you play this sport?': 'locations',
        'During which seasons would you be most likely to play this sport recreationally in Toronto?': 'seasons',
        'What types of music do you most associate with this sport?': 'music_genre',
        'What foods and drinks do you associate with watching this sport in the place or culture you know best?': 'food_drink',
        'What colour do you associate with this sport?': 'color',
        'Which song do you associate with this sport?': 'song',
        'How do you feel when you watch this sport?': 'emotions',
        'How much would you be willing to pay for a decent seat at a professional competition of this sport?': 'ticket_price'
    }
    df = df.rename(columns=column_mapping)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 2. Numeric & Ordinal Engineering
    price_regex = r'(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
    df['ticket_price'] = (
        df['ticket_price']
        .astype(str)
        .str.extract(price_regex)[0]
        .str.replace(',', '', regex=False)
    )
    df['ticket_price'] = pd.to_numeric(df['ticket_price'], errors='coerce')
    df['ticket_price'] = df['ticket_price'].clip(
        lower=price_cap_params['lower'],
        upper=price_cap_params['upper']
    )

    likert_mapping = {
        'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 
        'Agree': 4, 'Strongly agree': 5
    }
    for col in ['strategy_dependent', 'injury_risk', 'individual_brilliance']:
        df[col] = df[col].map(likert_mapping)

    # 3. Impute Saved Medians
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)

    # 4. Multi-select Encoding
    multiselect_cols = ['locations', 'seasons', 'music_genre']
    encoded_multiselect = []
    for col in multiselect_cols:
        temp_series = df[col].fillna('NULL_INDICATOR')
        dummies = temp_series.str.get_dummies(sep=',')
        dummies.columns = [f"{col}_{c.strip()}" for c in dummies.columns]
        encoded_multiselect.append(dummies)
        
    df = df.drop(columns=multiselect_cols)
    df = pd.concat([df] + encoded_multiselect, axis=1)

    # 5. TF-IDF Inference
    free_text_cols = ['food_drink', 'color', 'song', 'emotions']
    encoded_text_dfs = []
    for col in free_text_cols:
        saved_idf = tfidf_metadata.get(col, {})
        tfidf_df = compute_tfidf_inference(df[col], prefix=col, saved_idf=saved_idf)
        encoded_text_dfs.append(tfidf_df)
        
    df = df.drop(columns=free_text_cols)
    df = pd.concat([df] + encoded_text_dfs, axis=1)

    # 6. Align Features to Training State
    for expected_col in expected_features:
        if expected_col not in df.columns:
            df[expected_col] = 0.0
            
    df = df[expected_features].copy()

    # 7. Apply Saved Z-score Scaling
    for col in expected_features:
        mean_val = scaling_params[col]['mean']
        std_val = scaling_params[col]['std']
        df[col] = (df[col] - mean_val) / std_val

    return df.values.astype(np.float64)

def softmax_activation(z):
    """Numerically stable softmax."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict_softmax(X_scaled, weights_path='softmax_weights.json'):
    """Runs Softmax Regression forward pass using only NumPy."""
    data = load_json_safe(weights_path)

    weights = np.array(data['weights'], dtype=np.float64)
    biases = np.array(data['biases'], dtype=np.float64)
    classes = np.array(data['classes'])

    # weights has shape (n_classes, n_features) from sklearn's coef_
    z = np.dot(X_scaled, weights.T) + biases
    probs = softmax_activation(z)

    pred_indices = np.argmax(probs, axis=1)
    numeric_preds = classes[pred_indices]

    return [INT_TO_LABEL[int(p)] for p in numeric_preds]

def predict_all(filename):
    """Main entry point: loads raw CSV, preprocesses, and returns list of sport predictions."""
    if not os.path.exists(filename):
        print(f"[WARNING] Input CSV file not found: '{filename}'")
        raise FileNotFoundError(f"Missing input CSV file: {filename}")
        
    df = pd.read_csv(filename)
    X_scaled = preprocess_for_inference(df)
    return predict_softmax(X_scaled)

if __name__ == "__main__":
    input_file = input("Enter the path to the CSV file for prediction: ").strip()
    
    if not input_file:
        print("[WARNING] No file path entered. Exiting.")
        sys.exit(1)
        
    try:
        predictions = predict_all(input_file)
        df_input = pd.read_csv(input_file)
        
        if 'label' in df_input.columns:
            acc = np.mean(np.array(predictions) == df_input['label'].values)
            print(f"Verified Accuracy: {acc:.4f}")
        else:
            print(f"Generated {len(predictions)} predictions.")
            print(f"First 10 predictions: {predictions[:10]}")
            
    except FileNotFoundError:
        print("Prediction aborted due to missing file(s). Please verify your directory structure.")
        sys.exit(1)