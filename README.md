<div align="center">

# 🛡️ Anomaly Detection in Cloud Network Traffic
### Using Machine Learning — Random Forest + Isolation Forest

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.43-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Projet de Fin d'Année — École Nationale d'Ingénieurs de Tunis (ENIT)**
**Département Technologies de l'Information et de la Communication**
**Année universitaire : 2025/2026**

</div>

---

## Table des matières

- [À propos](#-à-propos)
- [Architecture](#-architecture)
- [Structure du projet](#-structure-du-projet)
- [Dataset](#-dataset)
- [Modèles](#-modèles)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Dashboard](#-dashboard)
- [Auteur](#-auteur)

---

## À propos

Ce projet propose un système de détection d'anomalies dans le trafic réseau cloud
basé sur l'apprentissage automatique. Il combine deux approches complémentaires :

- **Random Forest** (supervisé) — détecte les attaques connues avec haute précision
- **Isolation Forest** (non supervisé) — détecte les comportements anormaux inconnus
  sans jamais avoir vu les étiquettes d'attaque

Le système est déployé sous forme d'une **API REST FastAPI** et d'un
**tableau de bord interactif Streamlit** affichant les prédictions en temps réel.

---

## Architecture
┌─────────────────────────────────────────────┐
│  Dashboard Streamlit — 5 onglets            │
│  Simulation | CSV | Résultats | SHAP | Info │
└──────────────────┬──────────────────────────┘
│ HTTP REST
┌──────────────────▼──────────────────────────┐
│  API FastAPI — localhost:8000               │
│  /predict  /predict/batch  /stats  /ws/live │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│  Moteur ML                                  │
│  Random Forest (150 arbres, supervisé)      │
│  Isolation Forest (200 arbres, non supervisé│
│  StandardScaler — normalisation             │
└──────────────────┬──────────────────────────┘
│
┌──────────────────▼──────────────────────────┐
│  CICIDS2017 — 8 fichiers CSV — 2.8M flux   │
└─────────────────────────────────────────────┘

---

## Structure du projet
anomaly-detection-pfa/
│
├── dashboard/
│   └── app.py              # Interface Streamlit (5 onglets)
│
├── src/
│   ├── preprocess.py       # Nettoyage et normalisation des données
│   ├── train.py            # Entraînement RF + IF
│   ├── evaluate.py         # Évaluation et génération des figures
│   ├── shap_explain.py     # Explicabilité SHAP
│   ├── api.py              # Serveur FastAPI REST
│   ├── flow_builder.py     # Construction de flux réseau
│   ├── live_capture.py     # Capture réseau temps réel (Scapy)
│   ├── firewall.py         # Blocage automatique des IPs
│   └── monitor.py          # Moteur de surveillance réseau réelle
│
├── data/                   # ← Télécharger CICIDS2017 ici
├── models/                 # ← Générés par train.py
├── results/                # ← Générés par evaluate.py et shap_explain.py
│
├── requirements.txt
├── .gitignore
└── README.md
---

## Dataset

Nous utilisons le dataset **CICIDS2017** du Canadian Institute for Cybersecurity.

| Caractéristique | Valeur |
|---|---|
| Flux réseau totaux | 2 830 743 |
| Fichiers CSV | 8 |
| Features sélectionnées | 19 |
| Types d'attaques | 7 (DDoS, BruteForce, PortScan, Botnet...) |
| Ratio Normal / Attaque | ~80% / 20% |

**Télécharger le dataset :**
https://www.unb.ca/cic/datasets/ids-2017.html
ou
https://www.kaggle.com/datasets/cicdataset/cicids2017
Placer les fichiers CSV dans le dossier `data/`.

---

## Modèles

### Random Forest (Supervisé)

| Paramètre | Valeur |
|---|---|
| `n_estimators` | 150 |
| `max_depth` | 20 |
| `class_weight` | balanced |
| Labels requis | Oui |
| Attaques inconnues | Non |

### Isolation Forest (Non supervisé)

| Paramètre | Valeur |
|---|---|
| `n_estimators` | 200 |
| `contamination` | Ratio réel calculé |
| Labels requis | **Non** |
| Attaques inconnues | **Oui** |

---

## Installation

### Prérequis

- Python 3.11+
- Windows / Linux / macOS
- Npcap (Windows uniquement, pour la capture réseau)
  `https://npcap.com/#download`

### Installation

```bash
# Cloner le repo
git clone https://github.com/TON_USERNAME/anomaly-detection-pfa.git
cd anomaly-detection-pfa

# Créer l'environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux/macOS)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### 1. Télécharger le dataset

Placer les fichiers CSV de CICIDS2017 dans le dossier `data/`.

### 2. Entraîner les modèles

```bash
python src/train.py
```
Durée estimée : 10–15 minutes

### 3. Générer les figures d'évaluation

```bash
python src/evaluate.py
```

### 4. Générer les figures SHAP (optionnel)

```bash
python src/shap_explain.py
```

### 5. Lancer l'API (Terminal 1)

```bash
python src/api.py
```

API disponible sur : `http://localhost:8000`
Documentation Swagger : `http://localhost:8000/docs`

### 6. Lancer le Dashboard (Terminal 2)

```bash
cd dashboard
streamlit run app.py
```

Dashboard disponible sur : `http://localhost:8501`

### 7. Surveillance réseau réelle (optionnel — admin requis)

```bash
# Lister les interfaces
python src/monitor.py --list-interfaces

# Démarrer la surveillance
python src/monitor.py --interface "Wi-Fi"

# Sans blocage automatique
python src/monitor.py --interface "Wi-Fi" --no-block
```

---

## Résultats

| Métrique | Random Forest | Isolation Forest |
|---|---|---|
| F1-score (attaque) | ~0.98 | ~0.80 |
| AUC-ROC | ~0.99 | ~0.85 |
| Labels requis | Oui | **Non** |
| Détection zero-day | Non | **Oui** |

---

## Dashboard

Le tableau de bord Streamlit propose 5 onglets :

| Onglet | Description |
|---|---|
| Simulation temps réel | Graphique animé, jauge de risque, journal des alertes |
| Analyser un CSV | Upload fichier, prédiction instantanée, export résultats |
| Résultats du modèle | Figures ROC, confusion matrices, comparaison RF vs IF |
| Explication SHAP | Importance des features, beeswarm plots, comparaison |
| À propos | Architecture, paramètres, comparaison des approches |

---

## API Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Statut de l'API |
| `/health` | GET | Vérification de santé |
| `/features` | GET | Liste des 19 features |
| `/predict` | POST | Prédiction unitaire |
| `/predict/batch` | POST | Prédiction par lot |
| `/stats` | GET | Statistiques de session |
| `/ws/live` | WebSocket | Streaming temps réel |

---

##  Dépendances principales
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
fastapi==0.103.0
uvicorn==0.23.0
streamlit==1.27.0
plotly==5.17.0
shap==0.43.0
scapy
joblib==1.3.0