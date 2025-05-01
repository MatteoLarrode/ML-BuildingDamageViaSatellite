import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_at_thresholds(y_true, y_prob, thresholds):
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        precision_damaged = precision_score(y_true, y_pred, pos_label=1)
        recall_damaged = recall_score(y_true, y_pred, pos_label=1)
        f1_damaged = f1_score(y_true, y_pred, pos_label=1)
        
        precision_undamaged = precision_score(y_true, y_pred, pos_label=0)
        recall_undamaged = recall_score(y_true, y_pred, pos_label=0)
        f1_undamaged = f1_score(y_true, y_pred, pos_label=0)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Store results
        results.append({
            'threshold': threshold,
            'precision_damaged': precision_damaged * 100,
            'recall_damaged': recall_damaged * 100,
            'f1_damaged': f1_damaged * 100,
            'precision_undamaged': precision_undamaged * 100,
            'recall_undamaged': recall_undamaged * 100,
            'f1_undamaged': f1_undamaged * 100,
            'accuracy': accuracy * 100
        })
    
    return pd.DataFrame(results)