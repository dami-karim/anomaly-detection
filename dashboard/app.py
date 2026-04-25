import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import sys
import requests
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'src'
))
from preprocess import FEATURES

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Cloud Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.header-box {
    background: #0c447c;
    padding: 1.2rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
}
.header-box h1 {
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}
.header-box p {
    color: #B5D4F4;
    font-size: 0.82rem;
    margin: 0.2rem 0 0;
}
.kpi {
    background: var(--background-color);
    border: 0.5px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.kpi-label {
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.kpi-val {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.2rem 0 0;
}
.kpi-val.blue   { color: #185FA5; }
.kpi-val.red    { color: #E24B4A; }
.kpi-val.green  { color: #1D9E75; }
.kpi-val.orange { color: #BA7517; }
.sec {
    font-size: 0.95rem;
    font-weight: 600;
    color: #185FA5;
    border-left: 3px solid #185FA5;
    padding-left: 0.5rem;
    margin: 1rem 0 0.5rem;
}
.badge-a {
    background:#FCEBEB; color:#791F1F;
    padding:2px 8px; border-radius:5px;
    font-size:0.72rem; font-weight:600;
}
.badge-n {
    background:#EAF3DE; color:#27500A;
    padding:2px 8px; border-radius:5px;
    font-size:0.72rem; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# CHEMINS + CHARGEMENT MODÈLES
# ══════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
API         = "http://localhost:8000"


@st.cache_resource
def load_models():
    try:
        iso    = joblib.load(os.path.join(
            MODELS_DIR, 'isolation_forest.pkl'
        ))
        rf     = joblib.load(os.path.join(
            MODELS_DIR, 'random_forest.pkl'
        ))
        scaler = joblib.load(os.path.join(
            MODELS_DIR, 'scaler.pkl'
        ))
        return iso, rf, scaler
    except Exception:
        return None, None, None


iso, rf, scaler = load_models()

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
for k, v in [
    ('history', []),
    ('alerts',  []),
    ('idx',     0)
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════
st.markdown("""
<div class="header-box">
  <h1>Système de Détection d'Anomalies Réseau Cloud</h1>
  <p>
    Random Forest (supervisé) &nbsp;+&nbsp;
    Isolation Forest (non supervisé) &nbsp;|&nbsp;
    CICIDS2017 — 7 types d'attaques &nbsp;|&nbsp;
    Temps réel
  </p>
</div>
""", unsafe_allow_html=True)

# Vérification modèles
if iso is None:
    st.error(
        "Modèles non trouvés. "
        "Lance d'abord : python src/train.py"
    )
    st.stop()

# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Paramètres")
    threshold = st.slider(
        "Seuil de détection IF",
        0.0, 1.0, 0.5, 0.01
    )
    speed = st.slider(
        "Vitesse simulation (sec)",
        0.1, 2.0, 0.4, 0.1
    )
    st.divider()

    st.markdown("### Statut des modèles")
    st.success("Random Forest — chargé")
    st.success("Isolation Forest — chargé")

    try:
        r = requests.get(f"{API}/health", timeout=1)
        if r.status_code == 200:
            st.success("API — opérationnelle")
        else:
            st.warning("API — erreur")
    except Exception:
        st.error("API — non accessible")

    st.divider()
    st.markdown("### Dataset")
    st.info(
        "CICIDS2017 — complet\n\n"
        "7 types d'attaques :\n"
        "- DDoS\n"
        "- Brute Force\n"
        "- Port Scan\n"
        "- Botnet\n"
        "- Web Attacks\n"
        "- Infiltration\n"
        "- DoS"
    )

# ══════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Simulation temps réel",
    "Analyser un CSV",
    "Résultats du modèle",
    "Explication SHAP",
    "À propos",
    "Réseau réel"
])

# ══════════════════════════════════════════════════
# ONGLET 1 — SIMULATION TEMPS RÉEL
# ══════════════════════════════════════════════════
with tab1:

    @st.cache_data
    def load_sim_data():
        import glob
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        files    = glob.glob(
            os.path.join(DATA_DIR, '*.csv')
        )
        if not files:
            return None
        from preprocess import load_and_clean
        dfs = []
        for f in files[:2]:
            try:
                dfs.append(load_and_clean(f))
            except Exception:
                pass
        if not dfs:
            return None
        df    = pd.concat(dfs, ignore_index=True)
        avail = [
            feat for feat in FEATURES
            if feat in df.columns
        ]
        return df[avail + ['label']].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

    sim_df = load_sim_data()

    if sim_df is None:
        st.error(
            "Aucun CSV trouvé dans data/. "
            "Télécharge CICIDS2017."
        )
    else:
        avail_feat = [
            f for f in FEATURES
            if f in sim_df.columns
        ]

        # ── KPIs ──────────────────────────────────
        total   = len(st.session_state.history)
        attacks = sum(
            1 for h in st.session_state.history
            if h['is_attack']
        )
        rate   = round(attacks / max(total,1)*100, 1)
        rf_det = sum(
            1 for h in st.session_state.history
            if h.get('rf_prediction') == 1
        )
        if_det = sum(
            1 for h in st.session_state.history
            if h.get('iso_prediction') == 1
        )

        k1,k2,k3,k4,k5 = st.columns(5)
        for col, label, val, cls in zip(
            [k1,k2,k3,k4,k5],
            ['Flux analysés','Attaques',
             'Taux attaque','RF détecte','IF détecte'],
            [f'{total:,}', f'{attacks:,}',
             f'{rate}%', f'{rf_det:,}', f'{if_det:,}'],
            ['blue','red','red','blue','orange']
        ):
            col.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val {cls}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Layout graphiques ──────────────────────
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                '<p class="sec">'
                'Score IF en temps réel'
                '</p>',
                unsafe_allow_html=True
            )
            chart_ph = st.empty()
        with col2:
            st.markdown(
                '<p class="sec">Niveau de risque</p>',
                unsafe_allow_html=True
            )
            gauge_ph = st.empty()

        st.markdown(
            '<p class="sec">Journal des alertes</p>',
            unsafe_allow_html=True
        )
        alert_ph = st.empty()

        run = st.toggle(
            "Démarrer la simulation", value=False
        )

        # ── Fonctions graphiques ───────────────────
        def chart(hist):
            if not hist:
                fig = go.Figure()
                fig.update_layout(
                    height=280,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(
                        title='Score anomalie',
                        gridcolor='rgba(128,128,128,0.1)'
                    ),
                    annotations=[dict(
                        text='En attente de données...',
                        showarrow=False,
                        font=dict(size=14,
                                  color='#888780'),
                        xref='paper', yref='paper',
                        x=0.5, y=0.5
                    )]
                )
                return fig

            dh  = pd.DataFrame(hist[-100:])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=dh['iso_score'],
                mode='lines',
                line=dict(color='#378ADD', width=1.5),
                name='Score IF'
            ))
            fig.add_trace(go.Scatter(
                y=dh['iso_score'],
                mode='markers',
                marker=dict(
                    color=[
                        '#E24B4A' if a else '#1D9E75'
                        for a in dh['is_attack']
                    ],
                    size=6
                ),
                name='Statut',
                showlegend=False
            ))
            fig.add_hline(
                y=threshold,
                line_dash='dash',
                line_color='#EF9F27',
                annotation_text=f'Seuil {threshold}'
            )
            fig.update_layout(
                height=280,
                margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    title='Score anomalie',
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.02
                )
            )
            return fig

        def gauge(score):
            val = min(score * 100, 100)
            if val < 40:
                color = '#1D9E75'
                label = 'Normal'
            elif val < 70:
                color = '#EF9F27'
                label = 'Suspect'
            else:
                color = '#E24B4A'
                label = 'Danger'

            fig = go.Figure(go.Indicator(
                mode='gauge+number+delta',
                value=round(val, 1),
                number={'suffix': '%',
                        'font': {'size': 28}},
                delta={
                    'reference': 50,
                    'increasing': {
                        'color': '#E24B4A'
                    }
                },
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0,  40],
                         'color': '#EAF3DE'},
                        {'range': [40, 70],
                         'color': '#FAEEDA'},
                        {'range': [70, 100],
                         'color': '#FCEBEB'}
                    ],
                    'threshold': {
                        'line': {
                            'color': '#E24B4A',
                            'width': 3
                        },
                        'value': 70,
                        'thickness': 0.75
                    }
                },
                title={
                    'text': label,
                    'font': {'size': 14}
                }
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=10,r=10,t=30,b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig

        # ── RENDU PERMANENT ────────────────────────
        # Les graphiques sont TOUJOURS dessinés
        # depuis le session state
        # Ils persistent que la simulation tourne
        # ou non — c'est le fix principal
        last_score = (
            st.session_state.history[-1]['iso_score']
            if st.session_state.history
            else 0.0
        )

        chart_ph.plotly_chart(
            chart(st.session_state.history),
            use_container_width=True
        )
        gauge_ph.plotly_chart(
            gauge(last_score),
            use_container_width=True
        )
        if st.session_state.alerts:
            alert_ph.dataframe(
                pd.DataFrame(
                    st.session_state.alerts
                ),
                use_container_width=True
            )

        # ── BOUCLE SIMULATION ──────────────────────
        # Cette partie tourne seulement si run=True
        # Mais le rendu au-dessus est TOUJOURS actif
        if run:
            row = sim_df.iloc[
                st.session_state.idx % len(sim_df)
            ]
            try:
                res = requests.post(
                    f"{API}/predict",
                    json={
                        "features":
                            row[avail_feat].tolist()
                    },
                    timeout=2
                ).json()
                res['true_label'] = int(row['label'])
                st.session_state.history.append(res)

                if res['is_attack']:
                    st.session_state.alerts.insert(
                        0, {
                            'Flux #':
                                st.session_state.idx,
                            'Détecteur':
                                res['detector'],
                            'RF conf.':
                                res['rf_confidence'],
                            'IF score':
                                res['iso_score'],
                            'Vrai label': (
                                'Attaque'
                                if res['true_label']==1
                                else 'Normal'
                            )
                        }
                    )
                    st.session_state.alerts = \
                        st.session_state.alerts[:30]

            except Exception as e:
                st.error(f"API inaccessible : {e}")

            st.session_state.idx += 1
            time.sleep(speed)
            st.rerun()


# ══════════════════════════════════════════════════
# ONGLET 2 — ANALYSER UN CSV
# ══════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<p class="sec">'
        'Charger un fichier CSV à analyser'
        '</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Fichier CSV — format CICIDS2017",
        type=['csv']
    )

    if uploaded:
        with st.spinner("Lecture du fichier..."):
            try:
                df_up = pd.read_csv(
                    uploaded, encoding='utf-8',
                    low_memory=False
                )
            except UnicodeDecodeError:
                df_up = pd.read_csv(
                    uploaded, encoding='latin-1',
                    low_memory=False
                )
            df_up.columns = df_up.columns.str.strip()

        avail = [
            f for f in FEATURES
            if f in df_up.columns
        ]
        st.success(
            f"{len(df_up):,} lignes | "
            f"{len(avail)}/{len(FEATURES)} features"
        )

        if st.button(
            "Lancer l'analyse",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Analyse en cours..."):
                X = df_up[avail].replace(
                    [np.inf,-np.inf], np.nan
                ).dropna()
                Xs     = scaler.transform(X)
                scores = -iso.score_samples(Xs)
                preds  = (
                    iso.predict(Xs) == -1
                ).astype(int)

                rf_preds = rf.predict(Xs)
                rf_conf  = rf.predict_proba(Xs)[:,1]

                combined = (
                    (preds == 1) | (rf_preds == 1)
                ).astype(int)

            total_f  = len(X)
            n_attack = combined.sum()
            n_normal = total_f - n_attack

            k1,k2,k3 = st.columns(3)
            k1.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Flux analysés</div>
              <div class="kpi-val blue">
                {total_f:,}
              </div>
            </div>
            """, unsafe_allow_html=True)
            k2.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">
                Attaques détectées
              </div>
              <div class="kpi-val red">
                {n_attack:,}
              </div>
            </div>
            """, unsafe_allow_html=True)
            k3.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Taux</div>
              <div class="kpi-val red">
                {round(n_attack/total_f*100,1)}%
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=scores[combined==0],
                    name='Normal',
                    marker_color='#1D9E75',
                    opacity=0.7, nbinsx=50
                ))
                fig_h.add_trace(go.Histogram(
                    x=scores[combined==1],
                    name='Attaque',
                    marker_color='#E24B4A',
                    opacity=0.7, nbinsx=50
                ))
                fig_h.add_vline(
                    x=threshold,
                    line_dash='dash',
                    line_color='orange',
                    annotation_text='Seuil'
                )
                fig_h.update_layout(
                    barmode='overlay',
                    title='Distribution des scores',
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(
                    fig_h,
                    use_container_width=True
                )

            with c2:
                fig_p = go.Figure(go.Pie(
                    labels=['Normal','Attaque'],
                    values=[n_normal, n_attack],
                    marker_colors=[
                        '#1D9E75','#E24B4A'
                    ],
                    hole=0.45
                ))
                fig_p.update_layout(
                    title='Répartition',
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(
                    fig_p,
                    use_container_width=True
                )

            # Tableau
            res_df = pd.DataFrame({
                'Score IF':   np.round(scores, 4),
                'IF prédit':  [
                    'Attaque' if p else 'Normal'
                    for p in preds
                ],
                'RF prédit':  [
                    'Attaque' if p else 'Normal'
                    for p in rf_preds
                ],
                'RF conf.':   np.round(rf_conf, 4),
                'Décision':   [
                    'Attaque' if p else 'Normal'
                    for p in combined
                ]
            })
            st.markdown(
                '<p class="sec">Tableau détaillé</p>',
                unsafe_allow_html=True
            )
            st.dataframe(
                res_df,
                use_container_width=True
            )

            # Export
            st.download_button(
                "Télécharger les résultats CSV",
                res_df.to_csv(index=False),
                "predictions.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info(
            "Glisse un fichier CSV "
            "au format CICIDS2017."
        )

# ══════════════════════════════════════════════════
# ONGLET 3 — RÉSULTATS DU MODÈLE
# ══════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="sec">Figures générées '
        'par evaluate.py</p>',
        unsafe_allow_html=True
    )

    figures = [
        ('dashboard_summary.png',
         'Vue d\'ensemble complète'),
        ('roc_curves.png',
         'Courbes ROC — RF vs IF'),
        ('confusion_matrices.png',
         'Matrices de confusion'),
        ('comparison_rf_vs_if.png',
         'Comparaison des métriques'),
        ('anomaly_scores.png',
         'Distribution des scores IF'),
        ('feature_importance_rf.png',
         'Importance des features — RF'),
    ]

    any_found = False
    for fname, title in figures:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            any_found = True
            st.markdown(
                f'<p class="sec">{title}</p>',
                unsafe_allow_html=True
            )
            st.image(
                fpath,
                use_container_width=True
            )
            st.divider()

    if not any_found:
        st.warning(
            "Aucune figure trouvée. "
            "Lance : python src/evaluate.py"
        )
        st.code(
            "python src/evaluate.py",
            language="bash"
        )

# ══════════════════════════════════════════════════
# ONGLET 4 — SHAP
# ══════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<p class="sec">Explication SHAP — '
        'Isolation Forest</p>',
        unsafe_allow_html=True
    )
    st.caption(
        "SHAP révèle quelles features réseau "
        "ont le plus influencé chaque prédiction "
        "de l'Isolation Forest."
    )

    shap_figs = [
        ('shap_feature_importance.png',
         'Importance globale des features'),
        ('shap_summary.png',
         'SHAP Summary — beeswarm plot'),
    ]

    any_shap = False
    for fname, title in shap_figs:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            any_shap = True
            st.markdown(
                f'<p class="sec">{title}</p>',
                unsafe_allow_html=True
            )
            st.image(
                fpath,
                use_container_width=True
            )

    if not any_shap:
        st.warning(
            "Figures SHAP non générées. "
            "Lance : python src/shap_explain.py"
        )
        st.code(
            "python src/shap_explain.py",
            language="bash"
        )

# ══════════════════════════════════════════════════
# ONGLET 5 — À PROPOS
# ══════════════════════════════════════════════════
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<p class="sec">Architecture du système</p>',
            unsafe_allow_html=True
        )
        st.markdown("""
        **Couche 1 — Données**
        CICIDS2017 — tous les fichiers CSV
        chargés et combinés automatiquement.

        **Couche 2 — Modèles ML**
        - Random Forest (supervisé) — 150 arbres
        - Isolation Forest (non supervisé) — 200 arbres

        **Couche 3 — API**
        FastAPI sur localhost:8000
        Endpoints : /predict, /predict/batch, /stats

        **Couche 4 — Dashboard**
        Streamlit — ce tableau de bord
        5 onglets — temps réel + analyse CSV + résultats
        """)

        st.markdown(
            '<p class="sec">'
            'Isolation Forest — principe</p>',
            unsafe_allow_html=True
        )
        st.markdown("""
        L'Isolation Forest est entraîné **sans labels**.
        Il apprend seul quels flux sont anormaux
        en mesurant combien de coupures aléatoires
        sont nécessaires pour isoler chaque point.

        - Flux isolé rapidement → **anomalie**
        - Flux difficile à isoler → **normal**

        Avantage clé : peut détecter des attaques
        **inconnues** sans les avoir vues lors
        de l'entraînement.
        """)

    with c2:
        st.markdown(
            '<p class="sec">Paramètres des modèles</p>',
            unsafe_allow_html=True
        )
        params = pd.DataFrame({
            'Paramètre': [
                'RF — n_estimators',
                'RF — max_depth',
                'RF — class_weight',
                'IF — n_estimators',
                'IF — contamination',
                'IF — labels utilisés',
                'Dataset',
                'Features'
            ],
            'Valeur': [
                '150',
                '20',
                'balanced',
                '200',
                'ratio réel du dataset',
                'Non — jamais',
                'CICIDS2017 complet',
                f'{len(FEATURES)} features'
            ]
        })
        st.dataframe(
            params, hide_index=True,
            use_container_width=True
        )

        st.markdown(
            '<p class="sec">Comparaison RF vs IF</p>',
            unsafe_allow_html=True
        )
        comp = pd.DataFrame({
            'Critère': [
                'Type apprentissage',
                'Labels requis',
                'Attaques inconnues',
                'Précision connues',
                'Scalabilité cloud'
            ],
            'Random Forest': [
                'Supervisé', 'Oui',
                'Non', 'Très haute', 'Bonne'
            ],
            'Isolation Forest': [
                'Non supervisé', 'Non',
                'Oui', 'Modérée', 'Excellente'
            ]
        })
        st.dataframe(
            comp, hide_index=True,
            use_container_width=True
        )

        st.markdown(
            '<p class="sec">Liste des features</p>',
            unsafe_allow_html=True
        )
        feat_df = pd.DataFrame({
            'Feature': FEATURES,
            'Catégorie': [
                'Temporelle','Volumétrique',
                'Volumétrique','Volumétrique',
                'Volumétrique','Volumétrique',
                'Volumétrique','Débit',
                'Débit','Temporelle',
                'Temporelle','Temporelle',
                'Temporelle','Protocole TCP',
                'Protocole TCP','Protocole TCP',
                'Protocole TCP','Volumétrique',
                'Volumétrique'
            ]
        })
        st.dataframe(
            feat_df, hide_index=True,
            use_container_width=True
        )
# ══════════════════════════════════════════════════
# ONGLET 6 — RÉSEAU RÉEL
# ══════════════════════════════════════════════════
with tab6:
    import json
    from pathlib import Path

    ALERTS_FILE = os.path.join(
        BASE_DIR, 'results', 'live_alerts.json'
    )

    st.markdown(
        '<p class="sec">'
        'Surveillance réseau en temps réel'
        '</p>',
        unsafe_allow_html=True
    )

    # Instructions
    with st.expander(
        "Comment lancer la surveillance ?"
    ):
        st.markdown("""
        **1. Lancer en administrateur :**
```bash
        # Lister les interfaces disponibles
        python src/monitor.py --list-interfaces

        # Surveiller tout le trafic
        python src/monitor.py

        # Surveiller une IP spécifique
        python src/monitor.py --target 192.168.1.1

        # Sans blocage automatique
        python src/monitor.py --no-block

        # Avec seuil personnalisé
        python src/monitor.py --threshold 0.7
```

        **2. Npcap requis sur Windows :**
        `https://npcap.com/#download`
        """)

    # Lire le fichier d'alertes
    @st.fragment(run_every=2)
    def live_network_tab():
        if not Path(ALERTS_FILE).exists():
            st.info(
                "Aucune donnée. "
                "Lance monitor.py d'abord."
            )
            return

        try:
            with open(ALERTS_FILE) as f:
                data = json.load(f)
        except Exception:
            st.warning("Fichier alertes illisible.")
            return

        stats  = data.get("stats", {})
        alerts = data.get("alerts", [])

        # KPIs
        k1,k2,k3 = st.columns(3)
        k1.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">Flux analysés</div>
          <div class="kpi-val blue">
            {stats.get('total_flows', 0):,}
          </div>
        </div>
        """, unsafe_allow_html=True)
        k2.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">Attaques détectées</div>
          <div class="kpi-val red">
            {stats.get('total_attacks', 0):,}
          </div>
        </div>
        """, unsafe_allow_html=True)
        k3.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">IPs bloquées</div>
          <div class="kpi-val orange">
            {stats.get('blocked_ips', 0):,}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        if not alerts:
            st.success(
                "Aucune attaque détectée pour le moment."
            )
            return

        # Graphique timeline des alertes
        st.markdown(
            '<p class="sec">Timeline des attaques</p>',
            unsafe_allow_html=True
        )
        df_alerts = pd.DataFrame(alerts)
        if 'timestamp' in df_alerts.columns:
            df_alerts['timestamp'] = pd.to_datetime(
                df_alerts['timestamp']
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_alerts['timestamp'],
                y=df_alerts['iso_score'],
                mode='markers+lines',
                marker=dict(
                    color=[
                        '#E24B4A' if b
                        else '#EF9F27'
                        for b in df_alerts.get(
                            'blocked', []
                        )
                    ],
                    size=10
                ),
                line=dict(
                    color='#378ADD', width=1
                ),
                name='Score anomalie'
            ))
            fig.update_layout(
                height=280,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='Horodatage',
                yaxis_title='Score IF',
                margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

        # Tableau des alertes
        st.markdown(
            '<p class="sec">Détail des alertes</p>',
            unsafe_allow_html=True
        )
        display_cols = [
            'timestamp', 'src_ip', 'dst_ip',
            'detector', 'rf_conf',
            'iso_score', 'blocked'
        ]
        available_cols = [
            c for c in display_cols
            if c in df_alerts.columns
        ]
        st.dataframe(
            df_alerts[available_cols],
            use_container_width=True
        )

        # Section déblocage manuel
        st.divider()
        st.markdown(
            '<p class="sec">Déblocage manuel</p>',
            unsafe_allow_html=True
        )
        blocked_ips = df_alerts[
            df_alerts.get('blocked', False) == True
        ]['src_ip'].unique().tolist() \
            if 'blocked' in df_alerts.columns \
            else []

        if blocked_ips:
            ip_to_unblock = st.selectbox(
                "Choisir une IP à débloquer",
                blocked_ips
            )
            if st.button(
                f"Débloquer {ip_to_unblock}",
                type="secondary"
            ):
                st.warning(
                    f"Lance manuellement dans "
                    f"le terminal :\n\n"
                    f"```\nnetsh advfirewall "
                    f"firewall delete rule "
                    f"name=BLOCK_ANOMALY_"
                    f"{ip_to_unblock}\n```"
                )
        else:
            st.info("Aucune IP actuellement bloquée.")

    live_network_tab()