import threading
import time
import queue
from scapy.all import sniff, IP, TCP, UDP

from flow_builder import FlowBuilder


class LiveCapture:
    """
    Capture les paquets réseau en temps réel
    et les transforme en flux.
    """

    def __init__(self, interface=None,
                 target_ip=None):
        """
        interface  : nom de l'interface réseau
                     (None = auto-détection)
        target_ip  : si défini, surveille uniquement
                     le trafic vers/depuis cette IP
        """
        self.interface   = interface
        self.target_ip   = target_ip
        self.builder     = FlowBuilder()
        self.flow_queue  = queue.Queue()
        self.running     = False
        self._thread     = None
        self._exp_thread = None

    def start(self):
        """Démarre la capture dans un thread dédié."""
        self.running   = True
        self._thread   = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._exp_thread = threading.Thread(
            target=self._expiry_loop,
            daemon=True
        )
        self._thread.start()
        self._exp_thread.start()
        print(
            f"[CAPTURE] Démarré sur "
            f"{self.interface or 'auto'}"
            + (f" — cible : {self.target_ip}"
               if self.target_ip else "")
        )

    def stop(self):
        """Arrête la capture."""
        self.running = False
        print("[CAPTURE] Arrêté.")

    def get_flow(self, timeout=1.0):
        """
        Retourne un flux finalisé ou None
        si aucun flux disponible dans le délai.
        """
        try:
            return self.flow_queue.get(
                timeout=timeout
            )
        except queue.Empty:
            return None

    def _make_filter(self):
        """Construit le filtre BPF pour Scapy."""
        base = "ip and (tcp or udp)"
        if self.target_ip:
            return (
                f"ip and (tcp or udp) and "
                f"(host {self.target_ip})"
            )
        return base

    def _capture_loop(self):
        """Boucle principale de capture."""
        sniff(
            iface=self.interface,
            filter=self._make_filter(),
            prn=self._process_packet,
            store=False,
            stop_filter=lambda p: not self.running
        )

    def _expiry_loop(self):
        """
        Vérifie toutes les 10 secondes
        si des flux ont expiré (timeout).
        """
        while self.running:
            time.sleep(10)
            expired = self.builder.get_expired_flows()
            for flow in expired:
                if len(
                    flow.fwd_packets
                    + flow.bwd_packets
                ) >= 5:  # minimum 5 paquets
                    self.flow_queue.put(flow)

    def _process_packet(self, pkt):
        """Traite chaque paquet capturé."""
        if not (IP in pkt):
            return

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto  = pkt[IP].proto
        size   = len(pkt)

        src_port  = 0
        dst_port  = 0
        tcp_flags = None

        if TCP in pkt:
            src_port  = pkt[TCP].sport
            dst_port  = pkt[TCP].dport
            tcp_flags = str(pkt[TCP].flags)
            proto     = 6
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            proto    = 17

        completed = self.builder.process_packet(
            src_ip, src_port,
            dst_ip, dst_port,
            proto, size, tcp_flags
        )

        if completed and len(
            completed.fwd_packets
            + completed.bwd_packets
        ) >= 5:
            self.flow_queue.put(completed)