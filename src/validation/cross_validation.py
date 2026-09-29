from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def get_cross_validation_splits(df, target_col='is_climate_sensitive', group_col='location', n_splits=5, seed=42):
    """
    Génère un découpage StratifiedGroupKFold pour éviter le spatial data leakage.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    groups = df[group_col]
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        splits.append((train_idx, val_idx))
        print(f"Fold {fold+1} : Train={len(train_idx)}, Val={len(val_idx)}")
        
    return splits

if __name__ == "__main__":
    print("Module de validation croisée initialisé.")
