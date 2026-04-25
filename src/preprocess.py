import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Total',
    'Bwd IAT Total',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'Average Packet Size',
    'Avg Fwd Segment Size'
]

def load_and_clean(filepath):
    print(f"  Chargement : {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(
            filepath, encoding='utf-8',
            low_memory=False
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            filepath, encoding='latin-1',
            low_memory=False
        )

    df.columns = df.columns.str.strip()
    label_col  = 'Label'

    if label_col not in df.columns:
        raise ValueError(
            f"Colonne 'Label' absente dans "
            f"{os.path.basename(filepath)}"
        )

    available = [f for f in FEATURES
                 if f in df.columns]

    if not available:
        raise ValueError(
            f"Aucune feature reconnue dans "
            f"{os.path.basename(filepath)}"
        )

    df = df[available + [label_col]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['label'] = (
        df[label_col].str.strip() != 'BENIGN'
    ).astype(int)
    df.drop(columns=[label_col], inplace=True)

    n_normal  = (df['label'] == 0).sum()
    n_attack  = (df['label'] == 1).sum()
    print(f"    Normal : {n_normal:,} | "
          f"Attaque : {n_attack:,}")
    return df


def preprocess(df, fit=True, scaler_path=None):
    if scaler_path is None:
        base = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        scaler_path = os.path.join(
            base, 'models', 'scaler.pkl'
        )

    available = [f for f in FEATURES
                 if f in df.columns]
    X = df[available].copy()
    y = df['label'].values

    if fit:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(
            os.path.dirname(scaler_path),
            exist_ok=True
        )
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler sauvegardé : {scaler_path}")
    else:
        scaler   = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    return X_scaled, y