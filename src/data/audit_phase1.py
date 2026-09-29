import pandas as pd

def run_phase1_audit(data_dir="data/raw"):
    train = pd.read_csv(f"{data_dir}/Train.csv")
    climate = pd.read_csv(f"{data_dir}/climate_features.csv")
    
    print("=== AUDIT PHASE 1 ===")
    print(f"1. Lignes Train: {len(train)}, Lignes Climate: {len(climate)}")
    print(f"2. Valeurs manquantes Train: {train.isnull().sum().sum()}")
    print(f"3. Valeurs manquantes Climate: {climate.isnull().sum().sum()}")
    
    missing_ids = set(train['ID']) - set(climate['ID'])
    print(f"4. IDs Train manquants dans Climate: {len(missing_ids)}")
    
    merged = pd.merge(train[['ID', 'deathdate']], climate[['ID', 'deathdate']], on='ID', suffixes=('_train', '_climate'))
    date_match = (merged['deathdate_train'] == merged['deathdate_climate']).all()
    print(f"5. Correspondance exacte des dates : {date_match}")
    print("=== FIN DE L'AUDIT ===")

if __name__ == "__main__":
    run_phase1_audit()
