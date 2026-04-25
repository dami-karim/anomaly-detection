import shap
import joblib
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import glob

from preprocess import load_and_clean, preprocess, FEATURES

BASE_DIR    = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')


def explain():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Charger les données ─────────────────────
    print("Chargement des données...")
    csv_files = glob.glob(
        os.path.join(DATA_DIR, '*.csv')
    )

    if not csv_files:
        print(f"Erreur : Aucun CSV dans {DATA_DIR}")
        return

    dfs = []
    for f in csv_files:
        try:
            dfs.append(load_and_clean(f))
        except Exception as e:
            print(
                f"  Ignoré : "
                f"{os.path.basename(f)} — {e}"
            )

    if not dfs:
        print("Erreur : Aucun fichier chargé.")
        return

    df = pd.concat(dfs, ignore_index=True)
    X, y = preprocess(
        df, fit=False, scaler_path=SCALER_PATH
    )

    # ── 2. Charger les modèles ─────────────────────
    iso_path = os.path.join(
        MODELS_DIR, 'isolation_forest.pkl'
    )
    rf_path  = os.path.join(
        MODELS_DIR, 'random_forest.pkl'
    )

    if not os.path.exists(iso_path):
        print(
            "Erreur : isolation_forest.pkl introuvable.\n"
            "Lance d'abord : python src/train.py"
        )
        return

    iso = joblib.load(iso_path)
    print("Isolation Forest chargé.")

    rf_loaded = os.path.exists(rf_path)
    if rf_loaded:
        rf = joblib.load(rf_path)
        print("Random Forest chargé.")

    # ── 3. Échantillon pour SHAP ───────────────────
    np.random.seed(42)
    n_sample   = min(500, len(X))
    sample_idx = np.random.choice(
        len(X), n_sample, replace=False
    )
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]

    available = [f for f in FEATURES
                 if f in df.columns]

    print(f"\nÉchantillon : {n_sample} flux")
    print(f"Features    : {len(available)}")

    # ══════════════════════════════════════════════
    # SHAP — ISOLATION FOREST
    # ══════════════════════════════════════════════
    print("\nCalcul SHAP pour Isolation Forest...")
    print("(2 à 5 minutes selon ton CPU)")

    explainer_iso   = shap.TreeExplainer(iso)
    shap_values_iso = explainer_iso.shap_values(
        X_sample
    )

    # ── Figure 1 — Feature importance IF (bar) ────
    print("Génération figure 1 — IF feature importance")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_iso,
        X_sample,
        feature_names=available,
        plot_type="bar",
        show=False,
        color='#D85A30'
    )
    plt.title(
        "Importance des features — Isolation Forest\n"
        "Apprentissage non supervisé — SHAP Values",
        fontsize=13, fontweight='bold',
        pad=15
    )
    plt.xlabel(
        "Valeur SHAP moyenne "
        "(impact sur le score d'anomalie)",
        fontsize=11
    )
    plt.tight_layout()
    out1 = os.path.join(
        RESULTS_DIR, 'shap_feature_importance.png'
    )
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegardé : {out1}")

    # ── Figure 2 — Beeswarm IF ────────────────────
    print("Génération figure 2 — IF beeswarm")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_iso,
        X_sample,
        feature_names=available,
        show=False,
        plot_size=None,
        alpha=0.6
    )
    plt.title(
        "SHAP Summary — Isolation Forest\n"
        "Impact de chaque feature "
        "sur le score d'anomalie",
        fontsize=13, fontweight='bold',
        pad=15
    )
    plt.tight_layout()
    out2 = os.path.join(
        RESULTS_DIR, 'shap_summary.png'
    )
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegardé : {out2}")

    # ── Figure 3 — Explication flux le plus suspect
    print("Génération figure 3 — Flux le plus suspect")

    iso_scores      = -iso.score_samples(X_sample)
    most_suspicious = np.argmax(iso_scores)

    sv_single   = shap_values_iso[most_suspicious]
    sorted_idx  = np.argsort(
        np.abs(sv_single)
    )[::-1][:15]
    sorted_vals = sv_single[sorted_idx]
    sorted_feat = [available[i] for i in sorted_idx]

    colors = [
        '#E24B4A' if v > 0 else '#1D9E75'
        for v in sorted_vals
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        sorted_feat[::-1],
        sorted_vals[::-1],
        color=colors[::-1],
        alpha=0.85
    )
    ax.axvline(
        x=0, color='black',
        linewidth=0.8, linestyle='-'
    )
    ax.set_xlabel(
        "Valeur SHAP "
        "(contribution au score d'anomalie)",
        fontsize=11
    )
    ax.set_title(
        f"Explication — Flux le plus suspect\n"
        f"Score d'anomalie IF : "
        f"{iso_scores[most_suspicious]:.4f}\n"
        f"Rouge = pousse vers anomalie  |  "
        f"Vert = pousse vers normal",
        fontsize=12, fontweight='bold'
    )
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(
        ax.patches, sorted_vals[::-1]
    ):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=8,
            color='#333333'
        )

    plt.tight_layout()
    out3 = os.path.join(
        RESULTS_DIR, 'shap_waterfall_if.png'
    )
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegardé : {out3}")

    # ══════════════════════════════════════════════
    # SHAP — RANDOM FOREST (si disponible)
    # ══════════════════════════════════════════════
    if rf_loaded:
        print("\nCalcul SHAP pour Random Forest...")
        print("(1 à 3 minutes)")

        explainer_rf   = shap.TreeExplainer(rf)
        shap_values_rf = explainer_rf.shap_values(
            X_sample
        )

        sv_attack = (
            shap_values_rf[1]
            if isinstance(shap_values_rf, list)
            else shap_values_rf
        )

        # ── Figure 4 — RF feature importance ──────
        print(
            "Génération figure 4 — "
            "RF feature importance"
        )
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_attack,
            X_sample,
            feature_names=available,
            plot_type="bar",
            show=False,
            color='#378ADD'
        )
        plt.title(
            "Importance des features — Random Forest\n"
            "Classe 'Attaque' — SHAP Values",
            fontsize=13, fontweight='bold',
            pad=15
        )
        plt.xlabel(
            "Valeur SHAP moyenne "
            "(impact sur la prédiction d'attaque)",
            fontsize=11
        )
        plt.tight_layout()
        out4 = os.path.join(
            RESULTS_DIR,
            'shap_feature_importance_rf.png'
        )
        plt.savefig(
            out4, dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"Sauvegardé : {out4}")

        # ── Figure 5 — RF beeswarm ─────────────────
        print("Génération figure 5 — RF beeswarm")
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv_attack,
            X_sample,
            feature_names=available,
            show=False,
            plot_size=None,
            alpha=0.6
        )
        plt.title(
            "SHAP Summary — Random Forest\n"
            "Impact de chaque feature "
            "sur la détection d'attaque",
            fontsize=13, fontweight='bold',
            pad=15
        )
        plt.tight_layout()
        out5 = os.path.join(
            RESULTS_DIR, 'shap_summary_rf.png'
        )
        plt.savefig(
            out5, dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"Sauvegardé : {out5}")

        # ── Figure 6 — Comparaison IF vs RF ───────
        print(
            "Génération figure 6 — "
            "Comparaison IF vs RF"
        )

        mean_if = np.abs(
            shap_values_iso
        ).mean(axis=0)
        mean_rf = np.abs(sv_attack).mean(axis=0)

        mean_if_norm = mean_if / mean_if.max()
        mean_rf_norm = mean_rf / mean_rf.max()

        combined_importance = (
            mean_if_norm + mean_rf_norm
        )
        top10_idx   = np.argsort(
            combined_importance
        )[::-1][:10]
        top10_names = [
            available[i] for i in top10_idx
        ]

        x     = np.arange(len(top10_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))

        b1 = ax.bar(
            x - width/2,
            mean_if_norm[top10_idx],
            width,
            label='Isolation Forest (non supervisé)',
            color='#D85A30', alpha=0.85
        )
        b2 = ax.bar(
            x + width/2,
            mean_rf_norm[top10_idx],
            width,
            label='Random Forest (supervisé)',
            color='#378ADD', alpha=0.85
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            top10_names,
            rotation=35,
            ha='right',
            fontsize=9
        )
        ax.set_ylabel(
            'Importance SHAP normalisée',
            fontsize=11
        )
        ax.set_title(
            "Comparaison SHAP — IF vs RF\n"
            "Top 10 features les plus importantes",
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.bar_label(
            b1, fmt='%.2f',
            fontsize=8, padding=2
        )
        ax.bar_label(
            b2, fmt='%.2f',
            fontsize=8, padding=2
        )
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out6 = os.path.join(
            RESULTS_DIR,
            'shap_comparison_if_rf.png'
        )
        plt.savefig(
            out6, dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"Sauvegardé : {out6}")

    # ── Résumé final ───────────────────────────────
    print("\n" + "="*50)
    print("SHAP — Figures générées :")
    print("="*50)
    figures = [
        'shap_feature_importance.png',
        'shap_summary.png',
        'shap_waterfall_if.png',
    ]
    if rf_loaded:
        figures += [
            'shap_feature_importance_rf.png',
            'shap_summary_rf.png',
            'shap_comparison_if_rf.png'
        ]
    for f in figures:
        fpath  = os.path.join(RESULTS_DIR, f)
        status = (
            "OK" if os.path.exists(fpath)
            else "MANQUANT"
        )
        print(f"  [{status}] {f}")


if __name__ == '__main__':
    explain()