import joblib
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from preprocess import load_and_clean, preprocess
import glob
import pandas as pd

BASE_DIR    = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

BLUE   = '#378ADD'
RED    = '#E24B4A'
GREEN  = '#1D9E75'
ORANGE = '#EF9F27'
GRAY   = '#888780'


def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Charger tous les CSV
    csv_files = glob.glob(
        os.path.join(DATA_DIR, '*.csv')
    )
    dfs = [load_and_clean(f) for f in csv_files]
    df  = pd.concat(dfs, ignore_index=True)
    X, y = preprocess(
        df, fit=False, scaler_path=SCALER_PATH
    )

    rf  = joblib.load(os.path.join(
        MODELS_DIR, 'random_forest.pkl'
    ))
    iso = joblib.load(os.path.join(
        MODELS_DIR, 'isolation_forest.pkl'
    ))

    # Prédictions
    rf_pred   = rf.predict(X)
    rf_proba  = rf.predict_proba(X)[:, 1]
    rf_auc    = roc_auc_score(y, rf_proba)

    iso_pred  = (iso.predict(X) == -1).astype(int)
    iso_score = -iso.score_samples(X)
    iso_auc   = roc_auc_score(y, iso_score)

    rf_report  = classification_report(
        y, rf_pred, output_dict=True
    )
    iso_report = classification_report(
        y, iso_pred, output_dict=True
    )

    print("="*50)
    print("RANDOM FOREST (supervisé)")
    print("="*50)
    print(classification_report(
        y, rf_pred,
        target_names=['Normal', 'Attaque']
    ))
    print(f"AUC-ROC : {rf_auc:.4f}\n")

    print("="*50)
    print("ISOLATION FOREST (non supervisé)")
    print("="*50)
    print(classification_report(
        y, iso_pred,
        target_names=['Normal', 'Attaque']
    ))
    print(f"AUC-ROC : {iso_auc:.4f}\n")

    # ── Figure 1 — Courbes ROC ─────────────────────
    fpr_rf,  tpr_rf,  _ = roc_curve(y, rf_proba)
    fpr_iso, tpr_iso, _ = roc_curve(y, iso_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color=BLUE,
             linewidth=2,
             label=f'Random Forest — supervisé '
                   f'(AUC={rf_auc:.3f})')
    plt.plot(fpr_iso, tpr_iso, color=RED,
             linewidth=2,
             label=f'Isolation Forest — non supervisé '
                   f'(AUC={iso_auc:.3f})')
    plt.plot([0,1],[0,1], '--',
             color=GRAY, linewidth=1,
             label='Aléatoire (AUC=0.500)')
    plt.fill_between(fpr_rf, tpr_rf,
                     alpha=0.08, color=BLUE)
    plt.fill_between(fpr_iso, tpr_iso,
                     alpha=0.08, color=RED)
    plt.xlabel('Taux de Faux Positifs (FPR)',
               fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)',
               fontsize=12)
    plt.title('Courbes ROC — RF vs IF',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, 'roc_curves.png'),
        dpi=150
    )
    plt.close()
    print("Sauvegardé : roc_curves.png")

    # ── Figure 2 — Matrices de confusion ──────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Matrices de confusion — RF vs IF',
        fontsize=14, fontweight='bold'
    )
    for ax, pred, title, color in zip(
        axes,
        [rf_pred, iso_pred],
        ['Random Forest\n(supervisé)',
         'Isolation Forest\n(non supervisé)'],
        [BLUE, RED]
    ):
        cm = confusion_matrix(y, pred)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(title, fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Classe prédite', fontsize=11)
        ax.set_ylabel('Classe réelle', fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attaque'],
                           fontsize=10)
        ax.set_yticklabels(['Normal', 'Attaque'],
                           fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f'{cm[i,j]:,}',
                    ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white'
                    if cm[i,j] > cm.max()/2
                    else 'black'
                )
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR, 'confusion_matrices.png'
        ),
        dpi=150
    )
    plt.close()
    print("Sauvegardé : confusion_matrices.png")

    # ── Figure 3 — Distribution scores IF ─────────
    plt.figure(figsize=(9, 5))
    plt.hist(
        iso_score[y==0], bins=80,
        alpha=0.65, color=GREEN,
        label='Normal (BENIGN)',
        density=True
    )
    plt.hist(
        iso_score[y==1], bins=80,
        alpha=0.65, color=RED,
        label='Attaque',
        density=True
    )
    plt.axvline(
        x=np.percentile(iso_score, 95),
        color=ORANGE, linestyle='--',
        linewidth=2,
        label=f'Seuil 95e percentile'
    )
    plt.xlabel("Score d'anomalie — Isolation Forest",
               fontsize=12)
    plt.ylabel("Densité", fontsize=12)
    plt.title(
        "Distribution des scores d'anomalie\n"
        "L'IF sépare normal et attaques sans labels",
        fontsize=13, fontweight='bold'
    )
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR, 'anomaly_scores.png'
        ),
        dpi=150
    )
    plt.close()
    print("Sauvegardé : anomaly_scores.png")

    # ── Figure 4 — Comparaison métriques ──────────
    metrics  = ['Precision', 'Recall',
                'F1-score', 'AUC-ROC']
    rf_vals  = [
        rf_report['1']['precision'],
        rf_report['1']['recall'],
        rf_report['1']['f1-score'],
        rf_auc
    ]
    iso_vals = [
        iso_report['1']['precision'],
        iso_report['1']['recall'],
        iso_report['1']['f1-score'],
        iso_auc
    ]

    x     = np.arange(len(metrics))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, rf_vals, width,
                label='Random Forest (supervisé)',
                color=BLUE, alpha=0.85)
    b2 = ax.bar(x + width/2, iso_vals, width,
                label='Isolation Forest (non supervisé)',
                color=RED, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Comparaison des performances — RF vs IF',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt='%.3f',
                 fontsize=9, padding=2)
    ax.bar_label(b2, fmt='%.3f',
                 fontsize=9, padding=2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR, 'comparison_rf_vs_if.png'
        ),
        dpi=150
    )
    plt.close()
    print("Sauvegardé : comparison_rf_vs_if.png")

    # ── Figure 5 — Feature importance RF ──────────
    from preprocess import FEATURES
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    available   = [f for f in FEATURES
                   if f in df.columns]

    plt.figure(figsize=(10, 6))
    colors_bar = [
        BLUE if i < 5 else GRAY
        for i in range(len(available))
    ]
    plt.barh(
        [available[i] for i in indices],
        importances[indices],
        color=[colors_bar[i] for i in indices],
        alpha=0.85
    )
    plt.xlabel('Importance (Gini)',
               fontsize=12)
    plt.title(
        'Importance des features — Random Forest',
        fontsize=13, fontweight='bold'
    )
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR, 'feature_importance_rf.png'
        ),
        dpi=150
    )
    plt.close()
    print("Sauvegardé : feature_importance_rf.png")

    # ── Figure 6 — Dashboard résumé ───────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        'Tableau de bord — Résultats complets\n'
        'Détection d\'anomalies réseau cloud',
        fontsize=15, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.4, wspace=0.35
    )

    # ROC
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr_rf, tpr_rf, color=BLUE,
             linewidth=2,
             label=f'RF (AUC={rf_auc:.3f})')
    ax1.plot(fpr_iso, tpr_iso, color=RED,
             linewidth=2,
             label=f'IF (AUC={iso_auc:.3f})')
    ax1.plot([0,1],[0,1],'--',color=GRAY)
    ax1.set_title('Courbes ROC', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Métriques RF
    ax2 = fig.add_subplot(gs[0, 1])
    rf_m = [
        rf_report['1']['precision'],
        rf_report['1']['recall'],
        rf_report['1']['f1-score'],
        rf_auc
    ]
    bars = ax2.bar(
        ['Prec.','Recall','F1','AUC'],
        rf_m, color=BLUE, alpha=0.85
    )
    ax2.set_ylim(0, 1.1)
    ax2.set_title(
        'RF — métriques', fontweight='bold'
    )
    ax2.bar_label(bars, fmt='%.3f', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Métriques IF
    ax3 = fig.add_subplot(gs[0, 2])
    iso_m = [
        iso_report['1']['precision'],
        iso_report['1']['recall'],
        iso_report['1']['f1-score'],
        iso_auc
    ]
    bars3 = ax3.bar(
        ['Prec.','Recall','F1','AUC'],
        iso_m, color=RED, alpha=0.85
    )
    ax3.set_ylim(0, 1.1)
    ax3.set_title(
        'IF — métriques', fontweight='bold'
    )
    ax3.bar_label(bars3, fmt='%.3f', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Distribution IF
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(
        iso_score[y==0], bins=50,
        alpha=0.6, color=GREEN,
        label='Normal', density=True
    )
    ax4.hist(
        iso_score[y==1], bins=50,
        alpha=0.6, color=RED,
        label='Attaque', density=True
    )
    ax4.set_title(
        'Distribution scores IF',
        fontweight='bold'
    )
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # Feature importance top 8
    ax5 = fig.add_subplot(gs[1, 1:])
    top8 = indices[:8]
    ax5.barh(
        [available[i] for i in top8],
        importances[top8],
        color=BLUE, alpha=0.85
    )
    ax5.set_title(
        'Top 8 features importantes (RF)',
        fontweight='bold'
    )
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

    plt.savefig(
        os.path.join(
            RESULTS_DIR, 'dashboard_summary.png'
        ),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Sauvegardé : dashboard_summary.png")
    print("\nToutes les figures générées dans results/")


if __name__ == '__main__':
    evaluate()