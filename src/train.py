import numpy as np
import joblib
import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import (
    RandomForestClassifier, IsolationForest
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score
)
from preprocess import load_and_clean, preprocess

BASE_DIR    = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')


def train():
    # 1 — Charger tous les CSV
    csv_files = glob.glob(
        os.path.join(DATA_DIR, '*.csv')
    )
    if not csv_files:
        print(f"Erreur : Aucun CSV dans {DATA_DIR}")
        return

    print(f"\n{len(csv_files)} fichier(s) CSV trouvé(s)")
    dfs         = [load_and_clean(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nDataset combiné : {len(combined_df):,} flux")

    X, y = preprocess(
        combined_df, fit=True,
        scaler_path=SCALER_PATH
    )

    # 2 — Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    attack_ratio = np.clip(
        (y_train == 1).sum() / len(y_train),
        0.01, 0.5
    )
    print(f"\nRatio d'attaques : {attack_ratio:.4f}")

    # 3 — Random Forest
    print("\n" + "="*50)
    print("MODÈLE 1 — Random Forest (supervisé)")
    print("="*50)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(classification_report(
        y_test, rf_pred,
        target_names=['Normal', 'Attaque']
    ))

    # 4 — Isolation Forest
    print("\n" + "="*50)
    print("MODÈLE 2 — Isolation Forest (non supervisé)")
    print("="*50)
    iso = IsolationForest(
        n_estimators=200,
        contamination=attack_ratio,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train) # y_train absent — non supervisé

    iso_pred = (
        iso.predict(X_test) == -1
    ).astype(int)
    print(classification_report(
        y_test, iso_pred,
        target_names=['Normal', 'Attaque']
    ))
    print(f"F1  : {f1_score(y_test, iso_pred):.4f}")
    print(
        f"AUC : "
        f"{roc_auc_score(y_test, -iso.score_samples(X_test)):.4f}"
    )

    # 5 — Sauvegarder
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        rf,
        os.path.join(MODELS_DIR, 'random_forest.pkl')
    )
    joblib.dump(
        iso,
        os.path.join(
            MODELS_DIR, 'isolation_forest.pkl'
        )
    )
    print(f"\nModèles sauvegardés : {MODELS_DIR}")


if __name__ == '__main__':
    train()