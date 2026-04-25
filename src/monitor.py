import joblib
import numpy as np
import os
import sys
import json
import time
import threading
import platform
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from live_capture import LiveCapture
from firewall import Firewall
from preprocess import FEATURES

BASE_DIR   = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Fichier d'alertes partagé avec le dashboard
ALERTS_FILE = os.path.join(
    BASE_DIR, 'results', 'live_alerts.json'
)


class Monitor:
    """
    Moteur principal de surveillance réseau.
    Capture → Analyse → Alerte → Bloque.
    """

    def __init__(self,
                 interface=None,
                 target_ip=None,
                 auto_block=True,
                 threshold=0.6):
        """
        interface   : interface réseau à surveiller
        target_ip   : IP cible à surveiller
                      (None = tout le trafic)
        auto_block  : bloquer automatiquement
                      les IPs malveillantes
        threshold   : seuil score IF pour anomalie
        """
        self.interface  = interface
        self.target_ip  = target_ip
        self.auto_block = auto_block
        self.threshold  = threshold
        self.running    = False

        # Charger les modèles
        print("[MONITOR] Chargement des modèles...")
        self.iso    = joblib.load(os.path.join(
            MODELS_DIR, 'isolation_forest.pkl'
        ))
        self.rf     = joblib.load(os.path.join(
            MODELS_DIR, 'random_forest.pkl'
        ))
        self.scaler = joblib.load(os.path.join(
            MODELS_DIR, 'scaler.pkl'
        ))
        print("[MONITOR] Modèles chargés.")

        # Composants
        self.capture  = LiveCapture(
            interface=interface,
            target_ip=target_ip
        )
        self.firewall = Firewall()

        # Statistiques
        self.stats = {
            "total_flows":   0,
            "total_attacks": 0,
            "blocked_ips":   0,
            "start_time":    datetime.now()
                             .isoformat()
        }

        # Historique des alertes (en mémoire)
        self.alerts = []
        self._alerts_lock = threading.Lock()

        # Initialiser le fichier d'alertes
        os.makedirs(
            os.path.dirname(ALERTS_FILE),
            exist_ok=True
        )
        self._save_alerts()

    def start(self):
        """Démarre le monitoring."""
        self.running = True
        self.capture.start()
        print(
            f"[MONITOR] Surveillance démarrée\n"
            f"  Interface : "
            f"{self.interface or 'auto'}\n"
            f"  Cible     : "
            f"{self.target_ip or 'tout le trafic'}\n"
            f"  Seuil IF  : {self.threshold}\n"
            f"  Blocage   : "
            f"{'activé' if self.auto_block else 'désactivé'}\n"
        )
        self._analysis_loop()

    def stop(self):
        """Arrête le monitoring et débloque les IPs."""
        self.running = False
        self.capture.stop()
        if self.auto_block:
            self.firewall.unblock_all()
        print("[MONITOR] Arrêté.")

    def _analysis_loop(self):
        """Boucle principale d'analyse des flux."""
        while self.running:
            flow = self.capture.get_flow(timeout=1.0)
            if flow is None:
                continue

            try:
                features = flow.extract_features()
                result   = self._predict(features)

                self.stats["total_flows"] += 1

                if result["is_attack"]:
                    self.stats["total_attacks"] += 1
                    src_ip = flow.get_src_ip()
                    dst_ip = flow.get_dst_ip()
                    self._handle_attack(
                        result, src_ip, dst_ip, flow
                    )

            except Exception as e:
                print(f"[MONITOR] Erreur analyse : {e}")

    def _predict(self, features: list) -> dict:
        """Prédit si un flux est une attaque."""
        X  = np.array(features).reshape(1, -1)

        # Remplacer inf et NaN
        X = np.nan_to_num(
            X, nan=0.0,
            posinf=1e6, neginf=-1e6
        )

        Xs = self.scaler.transform(X)

        # Random Forest
        rf_pred  = int(self.rf.predict(Xs)[0])
        rf_proba = float(
            self.rf.predict_proba(Xs)[0][1]
        )

        # Isolation Forest
        iso_raw   = iso_pred = int(
            self.iso.predict(Xs)[0] == -1
        )
        iso_score = float(
            -self.iso.score_samples(Xs)[0]
        )

        # Décision combinée
        is_attack = bool(
            rf_pred == 1 or
            iso_score >= self.threshold
        )

        if rf_pred == 1 and iso_raw == 1:
            detector = "RF + IF"
        elif rf_pred == 1:
            detector = "RF"
        elif iso_raw == 1:
            detector = "IF"
        else:
            detector = "aucun"

        return {
            "rf_prediction":  rf_pred,
            "rf_confidence":  round(rf_proba, 4),
            "iso_prediction": iso_raw,
            "iso_score":      round(iso_score, 4),
            "is_attack":      is_attack,
            "detector":       detector
        }

    def _handle_attack(self, result,
                       src_ip, dst_ip, flow):
        """Gère une attaque détectée."""
        timestamp = datetime.now().isoformat()

        alert = {
            "timestamp":   timestamp,
            "src_ip":      src_ip,
            "dst_ip":      dst_ip,
            "protocol":    flow.get_protocol(),
            "rf_conf":     result["rf_confidence"],
            "iso_score":   result["iso_score"],
            "detector":    result["detector"],
            "blocked":     False,
            "fwd_packets": len(flow.fwd_packets),
            "bwd_packets": len(flow.bwd_packets),
        }

        # Bloquer l'IP source si auto_block activé
        if (self.auto_block
                and not self.firewall
                       ._is_local(src_ip)):
            blocked = self.firewall.block_ip(
                src_ip,
                reason=f"IF score={result['iso_score']}"
            )
            if blocked:
                alert["blocked"] = True
                self.stats["blocked_ips"] += 1

        # Ajouter l'alerte à la liste
        with self._alerts_lock:
            self.alerts.insert(0, alert)
            # Garder les 100 dernières alertes
            self.alerts = self.alerts[:100]
            self._save_alerts()

        # Afficher dans le terminal
        self._print_alert(alert)

    def _print_alert(self, alert):
        """Affiche l'alerte dans le terminal."""
        status = (
            "BLOQUÉE" if alert["blocked"]
            else "DÉTECTÉE"
        )
        print(
            f"\n{'='*55}\n"
            f"  ATTAQUE {status}\n"
            f"{'='*55}\n"
            f"  Horodatage  : {alert['timestamp']}\n"
            f"  Source IP   : {alert['src_ip']}\n"
            f"  Dest IP     : {alert['dst_ip']}\n"
            f"  Détecteur   : {alert['detector']}\n"
            f"  RF conf.    : {alert['rf_conf']}\n"
            f"  IF score    : {alert['iso_score']}\n"
            f"  Paquets     : "
            f"{alert['fwd_packets']} fwd / "
            f"{alert['bwd_packets']} bwd\n"
            f"{'='*55}\n"
        )

    def _save_alerts(self):
        """Sauvegarde les alertes dans un fichier JSON."""
        data = {
            "stats":  self.stats,
            "alerts": self.alerts
        }
        with open(ALERTS_FILE, 'w') as f:
            json.dump(data, f, indent=2,
                      default=str)

    def unblock_ip(self, ip: str):
        """Débloque manuellement une IP."""
        self.firewall.unblock_ip(ip)

    def get_blocked_ips(self):
        """Retourne la liste des IPs bloquées."""
        return list(self.firewall.blocked_ips)


def get_interfaces():
    """Liste les interfaces réseau disponibles."""
    try:
        from scapy.all import get_if_list
        return get_if_list()
    except Exception:
        return []


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Surveillance réseau temps réel'
    )
    parser.add_argument(
        '--interface', '-i',
        default=None,
        help='Interface réseau (ex: Ethernet, Wi-Fi)'
    )
    parser.add_argument(
        '--target', '-t',
        default=None,
        help='IP cible à surveiller'
    )
    parser.add_argument(
        '--no-block',
        action='store_true',
        help='Désactiver le blocage automatique'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Seuil score IF (0.0 à 1.0)'
    )
    parser.add_argument(
        '--list-interfaces',
        action='store_true',
        help='Lister les interfaces disponibles'
    )
    args = parser.parse_args()

    if args.list_interfaces:
        print("Interfaces disponibles :")
        for iface in get_interfaces():
            print(f"  - {iface}")
        exit(0)

    monitor = Monitor(
        interface=args.interface,
        target_ip=args.target,
        auto_block=not args.no_block,
        threshold=args.threshold
    )

    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n[MONITOR] Interruption clavier.")
        monitor.stop()