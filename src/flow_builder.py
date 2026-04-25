import time
import numpy as np
from collections import defaultdict
from threading import Lock


class Flow:
    """
    Représente un flux réseau bidirectionnel.
    Groupe les paquets par 5-tuple et calcule
    les 19 features utilisées par nos modèles.
    """

    TIMEOUT = 120  # secondes sans paquet = flux terminé

    def __init__(self, key):
        self.key          = key  # 5-tuple
        self.start_time   = time.time()
        self.last_time    = time.time()

        # Paquets forward (src→dst)
        self.fwd_packets  = []
        # Paquets backward (dst→src)
        self.bwd_packets  = []

        # Temps inter-arrivée (IAT)
        self.all_times    = []
        self.fwd_times    = []
        self.bwd_times    = []

        # Flags TCP
        self.syn_count    = 0
        self.rst_count    = 0
        self.psh_count    = 0
        self.ack_count    = 0

    def add_packet(self, size, is_forward,
                   tcp_flags=None):
        now = time.time()
        self.all_times.append(now)
        self.last_time = now

        if is_forward:
            self.fwd_packets.append(size)
            self.fwd_times.append(now)
        else:
            self.bwd_packets.append(size)
            self.bwd_times.append(now)

        if tcp_flags:
            if 'S' in tcp_flags:
                self.syn_count += 1
            if 'R' in tcp_flags:
                self.rst_count += 1
            if 'P' in tcp_flags:
                self.psh_count += 1
            if 'A' in tcp_flags:
                self.ack_count += 1

    def is_expired(self):
        return (time.time() - self.last_time
                > self.TIMEOUT)

    def _iat_stats(self, times):
        """Calcule mean et std des IAT"""
        if len(times) < 2:
            return 0.0, 0.0
        iats = [
            times[i+1] - times[i]
            for i in range(len(times)-1)
        ]
        return np.mean(iats), np.std(iats)

    def extract_features(self):
        """
        Extrait les 19 features dans le même ordre
        que FEATURES dans preprocess.py
        """
        duration = (
            (self.last_time - self.start_time)
            * 1e6  # en microsecondes
        )

        fwd_sizes = self.fwd_packets
        bwd_sizes = self.bwd_packets
        all_sizes = fwd_sizes + bwd_sizes

        total_fwd = len(fwd_sizes)
        total_bwd = len(bwd_sizes)
        total_len_fwd = sum(fwd_sizes)
        total_len_bwd = sum(bwd_sizes)
        total_len_all = total_len_fwd + total_len_bwd

        fwd_max = max(fwd_sizes) if fwd_sizes else 0
        bwd_max = max(bwd_sizes) if bwd_sizes else 0

        elapsed = max(duration / 1e6, 1e-6)
        bytes_per_sec   = total_len_all / elapsed
        packets_per_sec = len(all_sizes)   / elapsed

        iat_mean, iat_std = self._iat_stats(
            self.all_times
        )

        fwd_iat_total = (
            self.fwd_times[-1] - self.fwd_times[0]
            if len(self.fwd_times) > 1 else 0.0
        )
        bwd_iat_total = (
            self.bwd_times[-1] - self.bwd_times[0]
            if len(self.bwd_times) > 1 else 0.0
        )

        avg_pkt_size = (
            np.mean(all_sizes) if all_sizes else 0
        )
        avg_fwd_seg  = (
            np.mean(fwd_sizes) if fwd_sizes else 0
        )

        # Ordre identique à FEATURES dans preprocess.py
        return [
            duration,             # Flow Duration
            total_fwd,            # Total Fwd Packets
            total_bwd,            # Total Backward Packets
            total_len_fwd,        # Total Length of Fwd Packets
            total_len_bwd,        # Total Length of Bwd Packets
            fwd_max,              # Fwd Packet Length Max
            bwd_max,              # Bwd Packet Length Max
            bytes_per_sec,        # Flow Bytes/s
            packets_per_sec,      # Flow Packets/s
            iat_mean * 1e6,       # Flow IAT Mean (µs)
            iat_std  * 1e6,       # Flow IAT Std (µs)
            fwd_iat_total * 1e6,  # Fwd IAT Total (µs)
            bwd_iat_total * 1e6,  # Bwd IAT Total (µs)
            self.syn_count,       # SYN Flag Count
            self.rst_count,       # RST Flag Count
            self.psh_count,       # PSH Flag Count
            self.ack_count,       # ACK Flag Count
            avg_pkt_size,         # Average Packet Size
            avg_fwd_seg           # Avg Fwd Segment Size
        ]

    def get_src_ip(self):
        return self.key[0]

    def get_dst_ip(self):
        return self.key[2]

    def get_protocol(self):
        return self.key[4]


class FlowBuilder:
    """
    Maintient une table de flux actifs.
    Quand un flux expire ou reçoit un FIN/RST,
    il est finalisé et retourné pour prédiction.
    """

    def __init__(self):
        self.flows = {}
        self.lock  = Lock()

    def _make_key(self, src_ip, src_port,
                  dst_ip, dst_port, proto):
        """
        Crée une clé bidirectionnelle :
        le même flux dans les deux sens
        doit avoir la même clé
        """
        a = (src_ip, src_port, dst_ip,
             dst_port, proto)
        b = (dst_ip, dst_port, src_ip,
             src_port, proto)
        return min(a, b)

    def process_packet(self, src_ip, src_port,
                       dst_ip, dst_port,
                       proto, size,
                       tcp_flags=None):
        """
        Traite un paquet et retourne un flux
        finalisé si disponible, sinon None.
        """
        key = self._make_key(
            src_ip, src_port,
            dst_ip, dst_port, proto
        )

        completed_flow = None

        with self.lock:
            if key not in self.flows:
                self.flows[key] = Flow(key)

            flow      = self.flows[key]
            is_fwd    = (
                (src_ip, src_port) == (key[0], key[1])
            )
            flow.add_packet(size, is_fwd, tcp_flags)

            # Flux terminé si FIN ou RST
            if tcp_flags and (
                'F' in tcp_flags or
                'R' in tcp_flags
            ):
                completed_flow = flow
                del self.flows[key]

        return completed_flow

    def get_expired_flows(self):
        """
        Retourne et supprime les flux expirés
        (pas de paquet depuis TIMEOUT secondes)
        """
        expired = []
        with self.lock:
            expired_keys = [
                k for k, f in self.flows.items()
                if f.is_expired()
            ]
            for k in expired_keys:
                expired.append(self.flows.pop(k))
        return expired